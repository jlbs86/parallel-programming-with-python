"""Parallel dense Matrix Multiplication using mpi4py (row-wise decomposition).

Theory
------
Matrix multiplication C = A @ B is a classic data-parallel workload: each
row of C depends only on the corresponding row of A and the *entire*
matrix B. The standard distributed-memory strategy is:

1. Broadcast the full matrix B to every process (it's needed everywhere).
2. Scatter the rows of A across processes in contiguous blocks.
3. Each process computes its block of rows of C completely independently
   (no further communication -- this is the compute-heavy, embarrassingly
   parallel part).
4. Gather the row blocks of C back at the root in rank order.

This row-block decomposition is the same technique used by ScaLAPACK /
distributed BLAS implementations, just without the 2-D block-cyclic
tiling they use for better load balance at large scale.

Run with:
    mpiexec -n 4 python matrix_multiplication_parallel_mpi.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm
from numpy.typing import NDArray

MATRIX_DIM = 8  # chosen so it divides evenly for every size <= 8


class SequentialMatrixMultiplier:
    """Multiplies a local block of rows against a full matrix.

    :param rows: The local block of rows (a sub-matrix of ``A``).
    """

    def __init__(self, rows: NDArray[np.float64]) -> None:
        self._rows = rows

    def multiply(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute ``self._rows @ other``.

        :param other: The full right-hand-side matrix.
        :returns: The resulting block of rows of the product.
        """
        return self._rows @ other


@dataclass(frozen=True)
class MatrixMultiplicationResult:
    """Final result of a distributed row-wise matrix multiplication.

    :param a: The original left-hand-side matrix.
    :param b: The original right-hand-side matrix.
    :param product: The computed matrix product ``a @ b``.
    """

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    product: NDArray[np.float64]

    def is_correct(self) -> bool:
        """Verify the result against NumPy's own matrix multiplication.

        :returns: ``True`` if :attr:`product` is close to ``a @ b``.
        """
        return bool(np.allclose(self.product, self.a @ self.b))


class ParallelMatrixMultiplier:
    """Row-wise distributed matrix multiplication coordinator over MPI.

    :param comm: The MPI communicator whose ranks cooperate to compute the
        matrix product.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(
        self, a: NDArray[np.float64] | None, b: NDArray[np.float64] | None
    ) -> MatrixMultiplicationResult | None:
        """Compute ``a @ b`` using row-wise decomposition of ``a``.

        :param a: The left-hand-side matrix. Only meaningful on rank 0;
            every other rank should pass ``None``.
        :param b: The right-hand-side matrix. Only meaningful on rank 0;
            every other rank should pass ``None``.
        :returns: A :class:`MatrixMultiplicationResult` on rank 0, ``None``
            elsewhere.
        """
        if self._rank == 0:
            assert a is not None  # caller contract: only rank 0 passes real data
            row_chunks = self._split_into_row_chunks(a)
        else:
            row_chunks = None

        b = self._comm.bcast(b, root=0)
        my_rows = self._comm.scatter(row_chunks, root=0)

        local_result = SequentialMatrixMultiplier(my_rows).multiply(b)
        gathered = self._comm.gather(local_result, root=0)

        if self._rank != 0:
            return None

        assert a is not None  # guaranteed on rank 0, checked above
        assert gathered is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        return self._combine_results(a, b, gathered)

    def _split_into_row_chunks(self, a: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        """Split ``a`` into :attr:`_size` contiguous row blocks for scattering.

        :param a: The left-hand-side matrix.
        :returns: One contiguous row block per rank.
        """
        return np.array_split(a, self._size, axis=0)

    def _combine_results(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        gathered: list[NDArray[np.float64]],
    ) -> MatrixMultiplicationResult:
        """Stack per-rank row blocks of the product back into a full matrix.

        :param a: The original left-hand-side matrix.
        :param b: The original right-hand-side matrix.
        :param gathered: Per-rank row blocks of the product, in rank order.
        :returns: The combined :class:`MatrixMultiplicationResult`.
        """
        product = np.vstack(gathered)
        return MatrixMultiplicationResult(a=a, b=b, product=product)


def main() -> None:
    """Entry point: multiply two random matrices in parallel."""
    comm: Comm = MPI.COMM_WORLD
    multiplier = ParallelMatrixMultiplier(comm)

    a: NDArray[np.float64] | None = None
    b: NDArray[np.float64] | None = None
    if multiplier.rank == 0:
        rng = np.random.default_rng(2024)
        a = rng.integers(0, 10, size=(MATRIX_DIM, MATRIX_DIM)).astype("float64")
        b = rng.integers(0, 10, size=(MATRIX_DIM, MATRIX_DIM)).astype("float64")

    result = multiplier.run(a, b)

    if result is not None:
        print(f"Matrix A shape: {result.a.shape}, Matrix B shape: {result.b.shape}")
        print(f"Result matches sequential numpy @: {result.is_correct()}")


if __name__ == "__main__":
    main()
