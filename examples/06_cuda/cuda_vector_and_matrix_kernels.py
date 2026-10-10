"""GPU parallelism with CUDA in Python, via Numba's ``numba.cuda`` JIT backend.

Theory
------
CUDA exposes a different parallelism model from everything else in this
repository: instead of a handful of CPU cores (MPI processes / OpenMP
threads), a GPU runs **thousands of lightweight threads** organized in a
two-level hierarchy:

- A **grid** is launched for a kernel, made up of many **blocks**.
- Each **block** contains many **threads** that execute the same kernel
  code in lockstep groups of 32 (a "warp"), and can cooperate via fast
  on-chip **shared memory**.
- Thread/block indices (``cuda.grid(1)``, ``cuda.threadIdx``,
  ``cuda.blockIdx``) let each thread compute which slice of data it owns
  -- the GPU analogue of ``rank``/``size`` in MPI or the loop index handed
  out by :func:`numba.prange` on the CPU.

This is the right model for *massively* data-parallel, arithmetically
simple kernels (vector/matrix operations, image processing, N-body force
calculation) where you have far more independent work items than CPU
cores could ever provide -- exactly the kind of embarrassingly parallel
workload seen in ``monte_carlo_pi_parallel_mpi.py`` and
``matrix_multiplication_parallel_mpi.py``, but scaled from tens of CPU
cores to thousands of GPU cores.

``numba.cuda`` lets us write these kernels in Python syntax, JIT-compiled
to real CUDA PTX/SASS code by LLVM, instead of writing CUDA C/C++
directly.

.. important::
   Running this file requires an NVIDIA GPU, the CUDA toolkit, and a
   GPU-enabled Numba install (``pip install numba`` plus a working CUDA
   driver). See ``examples/08_gpu_setup_linux/`` in this repository for a
   full Linux GPU setup guide. On a machine without a CUDA-capable GPU
   (e.g. Apple Silicon, a cloud VM without a GPU), :func:`numba.cuda.is_available`
   will return ``False`` and :func:`main` will explain how to proceed
   instead of crashing.

Run with::

    python cuda_vector_and_matrix_kernels.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import cuda
from numpy.typing import NDArray

THREADS_PER_BLOCK = 256
THREADS_PER_BLOCK_2D = (16, 16)


@cuda.jit
def _vector_add_kernel(
    a: NDArray[np.float64], b: NDArray[np.float64], out: NDArray[np.float64]
) -> None:
    """Add two vectors elementwise, one GPU thread per element.

    Equivalent CUDA C kernel::

        __global__ void vectorAdd(const double *a, const double *b, double *out, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) out[i] = a[i] + b[i];
        }
    """
    i: int = cuda.grid(1)  # type: ignore[call-arg]  # numba stub mismatch for grid()
    if i < a.shape[0]:
        out[i] = a[i] + b[i]


@cuda.jit
def _matrix_multiply_kernel(
    a: NDArray[np.float64], b: NDArray[np.float64], out: NDArray[np.float64]
) -> None:
    """Dense matrix multiplication, one GPU thread per output element ``out[row, col]``.

    Each of the ``n * m`` output elements is independent (just like the
    row-decomposition in ``matrix_multiplication_parallel_mpi.py``, but
    decomposed all the way down to individual elements since the GPU has
    far more threads available than MPI has processes).
    """
    row, col = cuda.grid(2)  # type: ignore[call-arg,misc]  # numba stub mismatch for 2-D grid()
    if row < out.shape[0] and col < out.shape[1]:
        acc = 0.0
        for k in range(a.shape[1]):
            acc += a[row, k] * b[k, col]
        out[row, col] = acc


@dataclass(frozen=True)
class CudaKernelResult:
    """Result of running a CUDA kernel and validating it against NumPy.

    :param output: The array produced by the GPU kernel.
    :param matches_numpy: Whether the GPU result matches the equivalent
        NumPy computation within floating-point tolerance.
    """

    output: NDArray[np.float64]
    matches_numpy: bool


class CudaVectorAdder:
    """Launches the vector-addition kernel with a correctly sized grid.

    :param threads_per_block: Number of GPU threads per block.
    """

    def __init__(self, threads_per_block: int = THREADS_PER_BLOCK) -> None:
        self._threads_per_block = threads_per_block

    def run(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> CudaKernelResult:
        """Add ``a`` and ``b`` elementwise on the GPU.

        :param a: The first input vector.
        :param b: The second input vector, same shape as ``a``.
        :returns: A :class:`CudaKernelResult` with the GPU output.
        """
        n = a.shape[0]
        blocks_per_grid = math.ceil(n / self._threads_per_block)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_out = cuda.device_array_like(a)

        _vector_add_kernel[blocks_per_grid, self._threads_per_block](d_a, d_b, d_out)

        output = d_out.copy_to_host()
        return CudaKernelResult(output=output, matches_numpy=bool(np.allclose(output, a + b)))


class CudaMatrixMultiplier:
    """Launches the matrix-multiplication kernel with a correctly sized 2-D grid.

    :param threads_per_block: ``(x, y)`` GPU threads per block.
    """

    def __init__(self, threads_per_block: tuple[int, int] = THREADS_PER_BLOCK_2D) -> None:
        self._threads_per_block = threads_per_block

    def run(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> CudaKernelResult:
        """Multiply ``a @ b`` on the GPU.

        :param a: The left-hand matrix, shape ``(n, k)``.
        :param b: The right-hand matrix, shape ``(k, m)``.
        :returns: A :class:`CudaKernelResult` with the GPU output.
        """
        n, m = a.shape[0], b.shape[1]
        tx, ty = self._threads_per_block
        blocks_per_grid = (math.ceil(n / tx), math.ceil(m / ty))

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_out = cuda.device_array((n, m), dtype=np.float64)

        _matrix_multiply_kernel[blocks_per_grid, self._threads_per_block](d_a, d_b, d_out)

        output = d_out.copy_to_host()
        return CudaKernelResult(output=output, matches_numpy=bool(np.allclose(output, a @ b)))


def main() -> None:
    """Entry point: run both CUDA kernels if a GPU is available, else explain how to get one."""
    if not cuda.is_available():
        print("No CUDA-capable GPU detected on this machine.")
        print("This example requires an NVIDIA GPU with the CUDA toolkit installed.")
        print("See examples/08_gpu_setup_linux/README.md for a full Linux setup guide,")
        print("or run this script on a cloud GPU instance (e.g. a CUDA-enabled VM).")
        return

    rng = np.random.default_rng(0)

    vector_size = 1_000_000
    a = rng.uniform(-1, 1, size=vector_size)
    b = rng.uniform(-1, 1, size=vector_size)
    vector_result = CudaVectorAdder().run(a, b)
    print(f"[Vector add] matches NumPy: {vector_result.matches_numpy}")

    n = 512
    matrix_a = rng.uniform(-1, 1, size=(n, n))
    matrix_b = rng.uniform(-1, 1, size=(n, n))
    matrix_result = CudaMatrixMultiplier().run(matrix_a, matrix_b)
    print(f"[Matrix multiply] matches NumPy: {matrix_result.matches_numpy}")


if __name__ == "__main__":
    main()
