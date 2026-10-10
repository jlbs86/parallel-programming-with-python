"""Parallel 1-D Fast Fourier Transform using mpi4py (distributed butterfly).

Theory
------
The FFT recursively splits a signal into even- and odd-indexed samples
(Cooley-Tukey decimation-in-time), computes the FFT of each half, and
combines them with O(n) "twiddle factor" multiplications -- the butterfly
step. This recursive split/combine structure maps directly onto a
**recursive communicator-splitting** parallelization, very similar in
spirit to the parallel quicksort example in this repository:

1. The full signal starts entirely on rank 0 and is scattered: even
   samples go to the lower half of the process group, odd samples to the
   upper half (one round of point-to-point exchange within the current
   group, same idea as the hyperquicksort partner-exchange).
2. Each half of the process group recurses independently on its half of
   the communicator (``Comm.Split``) until a group has one process, which
   finishes with a local *sequential* FFT (recursive, in-process).
3. On the way back up the recursion, the two halves' local results are
   combined with the standard FFT butterfly formula using one
   ``sendrecv`` per pair of symmetric output indices.

This demonstrates how a classically "divide the *problem*" algorithm (FFT)
can be mapped onto "divide the *process group*" (the same pattern used by
parallel sorting networks), rather than the simpler scatter/gather used by
embarrassingly parallel examples elsewhere in this repo.

Run with:
    mpiexec -n 4 python fft_parallel_mpi.py
"""

from __future__ import annotations

import cmath
import random
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm


class SequentialFastFourierTransformer:
    """Local recursive Cooley-Tukey FFT, run once a process group has size 1.

    :param samples: The complex-valued signal to transform.
    """

    def __init__(self, samples: list[complex]) -> None:
        self._samples = samples

    def transform(self) -> list[complex]:
        """Recursively compute the FFT of :attr:`_samples`.

        :returns: The complex-valued frequency-domain representation.
        """
        return self._fft(self._samples)

    def _fft(self, samples: list[complex]) -> list[complex]:
        n = len(samples)
        if n <= 1:
            return samples

        even = self._fft(samples[0::2])
        odd = self._fft(samples[1::2])

        combined = [complex(0)] * n
        half = n // 2
        for k in range(half):
            twiddle = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
            combined[k] = even[k] + twiddle
            combined[k + half] = even[k] - twiddle
        return combined


@dataclass(frozen=True)
class FastFourierTransformResult:
    """Final result of a distributed FFT, available on rank 0.

    :param signal: The original input signal.
    :param spectrum: The computed frequency-domain representation.
    """

    signal: list[complex]
    spectrum: list[complex]

    def max_error_vs(self, expected: list[complex]) -> float:
        """Compute the maximum absolute error against a reference spectrum.

        :param expected: A reference spectrum (e.g. from ``numpy.fft.fft``).
        :returns: The maximum absolute difference between :attr:`spectrum`
            and ``expected``.
        """
        return max(abs(a - b) for a, b in zip(self.spectrum, expected))


class ParallelFastFourierTransformer:
    """Distributed FFT coordinator built on recursive communicator splitting.

    :param comm: The MPI communicator whose process group recurses in
        lockstep with the Cooley-Tukey divide-and-conquer recursion.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, signal: list[complex] | None) -> FastFourierTransformResult | None:
        """Compute the FFT of ``signal`` across the communicator.

        :param signal: The input signal. Only meaningful on rank 0; every
            other rank should pass ``None``.
        :returns: A :class:`FastFourierTransformResult` on rank 0, ``None``
            elsewhere.
        """
        signal = self._comm.bcast(signal, root=0)
        spectrum = self._parallel_fft(self._comm, signal)

        if self._rank != 0:
            return None

        return FastFourierTransformResult(signal=signal, spectrum=spectrum)

    def _parallel_fft(self, comm: Comm, samples: list[complex]) -> list[complex]:
        """Recursively bisect ``comm`` and ``samples`` until groups of size 1.

        :param comm: The (sub-)communicator for the current recursion level.
        :param samples: The samples local to this rank within ``comm``.
        :returns: The locally combined FFT result after full recursion.
        """
        size = comm.Get_size()
        rank = comm.Get_rank()
        n = len(samples)

        if size == 1 or n <= 1:
            return SequentialFastFourierTransformer(samples).transform()

        even = samples[0::2]
        odd = samples[1::2]

        half = size // 2
        lower_group = rank < half
        local_half = even if lower_group else odd

        color = 0 if lower_group else 1
        sub_comm = comm.Split(color=color, key=rank)
        try:
            local_result = self._parallel_fft(sub_comm, local_half)
        finally:
            sub_comm.Free()

        partner = rank + half if lower_group else rank - half
        partner_result = comm.sendrecv(local_result, dest=partner, source=partner)

        even_result = local_result if lower_group else partner_result
        odd_result = partner_result if lower_group else local_result

        return self._combine_halves(even_result, odd_result)

    def _combine_halves(
        self, even_result: list[complex], odd_result: list[complex]
    ) -> list[complex]:
        """Apply the FFT butterfly combination to two half-size results.

        :param even_result: The FFT of the even-indexed samples.
        :param odd_result: The FFT of the odd-indexed samples.
        :returns: The combined, full-size FFT result.
        """
        group_size = len(even_result) * 2
        combined = [complex(0)] * group_size
        for k in range(len(even_result)):
            twiddle = cmath.exp(-2j * cmath.pi * k / group_size) * odd_result[k]
            combined[k] = even_result[k] + twiddle
            combined[k + len(even_result)] = even_result[k] - twiddle
        return combined


def main() -> None:
    """Entry point: build a random signal on rank 0 and FFT it in parallel."""
    comm: Comm = MPI.COMM_WORLD
    transformer = ParallelFastFourierTransformer(comm)

    signal: list[complex] | None = None
    if transformer.rank == 0:
        random.seed(3)
        n = 16  # power of 2, required by this radix-2 Cooley-Tukey formulation
        signal = [complex(random.uniform(-1, 1), 0) for _ in range(n)]

    result = transformer.run(signal)

    if result is not None:
        try:
            import numpy as np

            expected = np.fft.fft(result.signal)
            max_error = result.max_error_vs(list(expected))
            print(f"Max difference vs numpy.fft.fft: {max_error:.2e}")
        except ImportError:
            pass
        print(f"FFT result (first 4 bins): {result.spectrum[:4]}")


if __name__ == "__main__":
    main()
