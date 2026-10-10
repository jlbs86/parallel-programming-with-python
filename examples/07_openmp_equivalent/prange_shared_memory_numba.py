"""OpenMP-equivalent shared-memory parallelism in Python using Numba's ``prange``.

Theory
------
OpenMP is a C/C++/Fortran API for shared-memory parallelism: a single
process spawns multiple **threads** that all see the same memory, and a
compiler directive like ``#pragma omp parallel for`` tells the compiler to
split a loop's iterations across threads automatically.

Pure Python cannot do this directly for CPU-bound code because of the
Global Interpreter Lock (GIL): plain Python threads cannot run bytecode
truly concurrently on multiple cores. `Numba <https://numba.pydata.org/>`_
closes this gap by JIT-compiling a Python function to native machine code
with LLVM and, when given ``parallel=True``, releasing the GIL and
distributing the iterations of a :func:`numba.prange` loop across OS
threads -- functionally the same contract as ``#pragma omp parallel for``:

- :func:`numba.njit` (``nopython`` mode) compiles the function to native
  code with no Python object overhead, the same way ``gcc -O3`` compiles C.
- ``parallel=True`` enables Numba's automatic parallelization pass,
  which detects data-parallel loops and farms them out to a thread pool
  sized by :data:`numba.config.NUMBA_NUM_THREADS` (equivalent to OpenMP's
  ``OMP_NUM_THREADS``).
- :func:`numba.prange` ("parallel range") explicitly marks a loop as safe
  to split across threads, the direct analogue of ``#pragma omp parallel
  for`` -- each iteration must be independent (no cross-iteration data
  dependencies), exactly like an OpenMP parallel loop.

This file demonstrates the pattern on three classic data-parallel kernels:
elementwise array transformation, a parallel reduction (sum), and a dense
matrix multiplication -- the same algorithm family used by the MPI
``examples/05_parallel_algorithms/matrix_multiplication_parallel_mpi.py``
example, but using **threads sharing one process's memory** instead of
**separate processes passing messages**.

Run with::

    python prange_shared_memory_numba.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numba.np.ufunc.parallel import get_thread_count
from numpy.typing import NDArray


@dataclass(frozen=True)
class BenchmarkResult:
    """Wall-clock comparison between a sequential and a ``prange`` kernel.

    :param sequential_seconds: Time taken by the plain NumPy/Python version.
    :param parallel_seconds: Time taken by the ``@njit(parallel=True)`` version.
    :param results_match: Whether both versions produced numerically equal output.
    """

    sequential_seconds: float
    parallel_seconds: float
    results_match: bool

    @property
    def speedup(self) -> float:
        """Ratio of sequential to parallel runtime (>1 means the parallel version won).

        :returns: ``sequential_seconds / parallel_seconds``.
        """
        return self.sequential_seconds / self.parallel_seconds


@njit(cache=True)
def _sequential_elementwise_transform(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply a nonlinear transform to every element, one at a time (``range``)."""
    result = np.empty_like(values)
    for i in range(values.shape[0]):
        result[i] = np.sin(values[i]) ** 2 + np.cos(values[i]) ** 2
    return result


@njit(parallel=True, cache=True)
def _parallel_elementwise_transform(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the same transform, but split across threads with ``prange``.

    Equivalent OpenMP::

        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            result[i] = sin(values[i]) * sin(values[i]) + cos(values[i]) * cos(values[i]);
        }
    """
    result = np.empty_like(values)
    for i in prange(values.shape[0]):
        result[i] = np.sin(values[i]) ** 2 + np.cos(values[i]) ** 2
    return result


@njit(cache=True)
def _sequential_reduction_sum(values: NDArray[np.float64]) -> float:
    """Sum every element sequentially."""
    total = 0.0
    for i in range(values.shape[0]):
        total += values[i]
    return total


@njit(parallel=True, cache=True)
def _parallel_reduction_sum(values: NDArray[np.float64]) -> float:
    """Sum every element with a ``prange`` reduction.

    Numba recognizes the accumulation pattern ``total += ...`` inside a
    ``prange`` loop and automatically generates a *parallel reduction*:
    each thread accumulates a private partial sum, and the partials are
    combined at the end -- the same transformation OpenMP performs for
    ``#pragma omp parallel for reduction(+:total)``.
    """
    total = 0.0
    for i in prange(values.shape[0]):
        total += values[i]
    return total


@njit(cache=True)
def _sequential_matmul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dense matrix multiplication with triple-nested sequential loops."""
    n, k = a.shape
    _, m = b.shape
    result = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for p in range(k):
                acc += a[i, p] * b[p, j]
            result[i, j] = acc
    return result


@njit(parallel=True, cache=True)
def _parallel_matmul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dense matrix multiplication with the outer loop parallelized via ``prange``.

    Only the outermost loop (over output rows ``i``) is marked ``prange``:
    each row of the result depends only on row ``i`` of ``a`` and the full
    matrix ``b``, so rows are independent and safe to compute concurrently
    -- the same row-decomposition strategy used by the MPI
    ``matrix_multiplication_parallel_mpi.py`` example, but via threads
    sharing memory instead of processes exchanging messages.
    """
    n, k = a.shape
    _, m = b.shape
    result = np.zeros((n, m))
    for i in prange(n):
        for j in range(m):
            acc = 0.0
            for p in range(k):
                acc += a[i, p] * b[p, j]
            result[i, j] = acc
    return result


class SharedMemoryBenchmarkRunner:
    """Runs sequential vs. ``prange``-parallel kernels and reports timing.

    :param warmup: Whether to call each kernel once before timing, to
        exclude Numba's JIT compilation overhead from the measurement.
    """

    def __init__(self, warmup: bool = True) -> None:
        self._warmup = warmup

    def benchmark_elementwise_transform(self, size: int) -> BenchmarkResult:
        """Compare sequential vs. parallel elementwise transform on ``size`` elements.

        :param size: The number of elements in the input array.
        :returns: A :class:`BenchmarkResult` comparing both kernels.
        """
        values = np.random.default_rng(0).uniform(-np.pi, np.pi, size=size)
        if self._warmup:
            _sequential_elementwise_transform(values[:8])
            _parallel_elementwise_transform(values[:8])

        start = time.perf_counter()
        sequential_result = _sequential_elementwise_transform(values)
        sequential_seconds = time.perf_counter() - start

        start = time.perf_counter()
        parallel_result = _parallel_elementwise_transform(values)
        parallel_seconds = time.perf_counter() - start

        return BenchmarkResult(
            sequential_seconds=sequential_seconds,
            parallel_seconds=parallel_seconds,
            results_match=bool(np.allclose(sequential_result, parallel_result)),
        )

    def benchmark_reduction_sum(self, size: int) -> BenchmarkResult:
        """Compare sequential vs. parallel reduction-sum on ``size`` elements.

        :param size: The number of elements in the input array.
        :returns: A :class:`BenchmarkResult` comparing both kernels.
        """
        values = np.random.default_rng(1).uniform(0, 1, size=size)
        if self._warmup:
            _sequential_reduction_sum(values[:8])
            _parallel_reduction_sum(values[:8])

        start = time.perf_counter()
        sequential_result = _sequential_reduction_sum(values)
        sequential_seconds = time.perf_counter() - start

        start = time.perf_counter()
        parallel_result = _parallel_reduction_sum(values)
        parallel_seconds = time.perf_counter() - start

        return BenchmarkResult(
            sequential_seconds=sequential_seconds,
            parallel_seconds=parallel_seconds,
            results_match=bool(np.isclose(sequential_result, parallel_result)),
        )

    def benchmark_matmul(self, n: int) -> BenchmarkResult:
        """Compare sequential vs. parallel ``n x n`` matrix multiplication.

        :param n: The dimension of the square matrices to multiply.
        :returns: A :class:`BenchmarkResult` comparing both kernels.
        """
        rng = np.random.default_rng(2)
        a = rng.uniform(-1, 1, size=(n, n))
        b = rng.uniform(-1, 1, size=(n, n))
        if self._warmup:
            _sequential_matmul(a[:4, :4], b[:4, :4])
            _parallel_matmul(a[:4, :4], b[:4, :4])

        start = time.perf_counter()
        sequential_result = _sequential_matmul(a, b)
        sequential_seconds = time.perf_counter() - start

        start = time.perf_counter()
        parallel_result = _parallel_matmul(a, b)
        parallel_seconds = time.perf_counter() - start

        return BenchmarkResult(
            sequential_seconds=sequential_seconds,
            parallel_seconds=parallel_seconds,
            results_match=bool(np.allclose(sequential_result, parallel_result)),
        )


def main() -> None:
    """Entry point: run all three benchmarks and print a speedup summary."""
    print(f"Numba thread pool size (like OMP_NUM_THREADS): {get_thread_count()}")

    runner = SharedMemoryBenchmarkRunner()

    transform_result = runner.benchmark_elementwise_transform(size=20_000_000)
    print(
        f"\n[Elementwise transform] sequential={transform_result.sequential_seconds:.4f}s "
        f"parallel={transform_result.parallel_seconds:.4f}s "
        f"speedup={transform_result.speedup:.2f}x "
        f"match={transform_result.results_match}"
    )

    reduction_result = runner.benchmark_reduction_sum(size=50_000_000)
    print(
        f"[Reduction sum]        sequential={reduction_result.sequential_seconds:.4f}s "
        f"parallel={reduction_result.parallel_seconds:.4f}s "
        f"speedup={reduction_result.speedup:.2f}x "
        f"match={reduction_result.results_match}"
    )

    matmul_result = runner.benchmark_matmul(n=600)
    print(
        f"[Matrix multiplication] sequential={matmul_result.sequential_seconds:.4f}s "
        f"parallel={matmul_result.parallel_seconds:.4f}s "
        f"speedup={matmul_result.speedup:.2f}x "
        f"match={matmul_result.results_match}"
    )
    print(
        "\nNote: matmul speedup depends heavily on matrix size and cache "
        "behavior -- for small matrices, thread-launch overhead can exceed "
        "the compute saved, exactly as with OpenMP's `#pragma omp parallel "
        "for` on short loops. Larger matrices amortize that overhead."
    )


if __name__ == "__main__":
    main()
