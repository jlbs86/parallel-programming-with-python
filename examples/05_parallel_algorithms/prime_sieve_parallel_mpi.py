"""Parallel Prime Sieve using mpi4py (segmented Sieve of Eratosthenes).

Theory
------
The Sieve of Eratosthenes finds all primes up to N by crossing out
multiples of each prime. The classic distributed-memory parallelization is
the **segmented sieve**:

1. Rank 0 sequentially sieves the small base range [2, sqrt(N)] -- this
   part is cheap and inherently sequential, so it is not worth
   parallelizing.
2. The base primes are broadcast to every rank (``bcast``).
3. The remaining range (sqrt(N), N] is split into contiguous, equally sized
   segments, one per rank (**data decomposition** of the search space).
4. Each rank independently crosses out multiples of every base prime
   within its own segment only -- no communication needed during this
   compute-heavy phase.
5. Surviving numbers (the primes found) in each segment are gathered back
   at rank 0 in rank order and concatenated.

This is the same decomposition strategy used by large-scale primality
search projects (e.g. GIMPS-style segmented sieving) to sieve ranges that
do not fit in a single machine's memory.

Run with:
    mpiexec -n 4 python prime_sieve_parallel_mpi.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

N = 1_000_000


class SequentialPrimeSieve:
    """Sequential Sieve of Eratosthenes used for the base range and segments.

    :param limit: The inclusive upper bound to sieve up to, when sieving a
        base range starting at 2.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit

    def sieve_base_primes(self) -> list[int]:
        """Sieve all primes in ``[2, limit]``.

        :returns: The sorted list of primes up to :attr:`_limit`.
        """
        is_prime = [True] * (self._limit + 1)
        is_prime[0:2] = [False, False]
        for i in range(2, int(math.sqrt(self._limit)) + 1):
            if is_prime[i]:
                for multiple in range(i * i, self._limit + 1, i):
                    is_prime[multiple] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    def sieve_segment(self, start: int, end: int, base_primes: list[int]) -> list[int]:
        """Sieve the inclusive range ``[start, end]`` using known base primes.

        :param start: The inclusive start of the segment.
        :param end: The inclusive end of the segment.
        :param base_primes: Primes up to ``sqrt(end)``, used to cross out
            composites within the segment.
        :returns: The primes found within ``[start, end]``.
        """
        size = end - start + 1
        is_prime = [True] * size

        for prime in base_primes:
            # First multiple of `prime` that falls inside [start, end].
            first_multiple = max(prime * prime, ((start + prime - 1) // prime) * prime)
            for multiple in range(first_multiple, end + 1, prime):
                is_prime[multiple - start] = False

        return [start + i for i, prime in enumerate(is_prime) if prime]


@dataclass(frozen=True)
class PrimeSieveResult:
    """Final result of a distributed segmented sieve, available on rank 0.

    :param limit: The upper bound primes were sieved up to.
    :param primes: The sorted list of all primes found.
    """

    limit: int
    primes: list[int]


class ParallelPrimeSieveCoordinator:
    """Distributed segmented sieve coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to sieve
        disjoint segments of the search range.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, limit: int) -> PrimeSieveResult | None:
        """Sieve all primes up to ``limit`` across the communicator.

        :param limit: The inclusive upper bound to sieve up to.
        :returns: A :class:`PrimeSieveResult` on rank 0, ``None`` elsewhere.
        """
        base_limit = int(math.sqrt(limit)) + 1

        if self._rank == 0:
            base_primes = SequentialPrimeSieve(base_limit).sieve_base_primes()
        else:
            base_primes = None

        base_primes = self._comm.bcast(base_primes, root=0)

        my_start, my_end = self._segment_bounds(base_limit, limit)
        local_primes = SequentialPrimeSieve(limit).sieve_segment(my_start, my_end, base_primes)
        gathered = self._comm.gather(local_primes, root=0)

        if self._rank != 0:
            return None

        assert gathered is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        return self._combine_results(limit, base_primes, gathered)

    def _segment_bounds(self, base_limit: int, limit: int) -> tuple[int, int]:
        """Compute this rank's contiguous segment of the search range.

        :param base_limit: The upper bound of the sequentially-sieved base range.
        :param limit: The overall inclusive upper bound to sieve up to.
        :returns: A tuple ``(start, end)`` describing this rank's segment.
        """
        segment_length = (limit - base_limit) // self._size
        my_start = base_limit + 1 + self._rank * segment_length
        my_end = limit if self._rank == self._size - 1 else my_start + segment_length - 1
        return my_start, my_end

    def _combine_results(
        self, limit: int, base_primes: list[int], gathered: list[list[int]]
    ) -> PrimeSieveResult:
        """Combine the base primes with every rank's segment primes.

        :param limit: The overall inclusive upper bound sieved.
        :param base_primes: Primes found in the sequentially-sieved base range.
        :param gathered: Per-rank segment primes, in rank order.
        :returns: The combined :class:`PrimeSieveResult`.
        """
        all_primes = base_primes + [p for group in gathered for p in group]
        return PrimeSieveResult(limit=limit, primes=all_primes)


def main() -> None:
    """Entry point: sieve all primes up to :data:`N` in parallel."""
    comm: Comm = MPI.COMM_WORLD
    coordinator = ParallelPrimeSieveCoordinator(comm)

    result = coordinator.run(N)

    if result is not None:
        print(f"Found {len(result.primes)} primes up to {result.limit}")
        print(f"Largest 5: {result.primes[-5:]}")


if __name__ == "__main__":
    main()
