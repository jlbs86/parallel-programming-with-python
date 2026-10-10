"""Parallel Monte Carlo estimation of Pi using mpi4py (embarrassingly parallel).

Theory
------
Monte Carlo Pi estimation samples random points in a unit square and counts
how many land inside the inscribed unit circle; the ratio approximates
pi/4. Each sample is fully independent of every other sample, making this
the textbook example of an **embarrassingly parallel** workload:

1. Every rank generates and tests its own share of random samples locally,
   with a per-rank random seed (no communication during the heavy
   computation at all).
2. Only a single scalar (the local hit count) needs to be combined, which
   is done with one ``reduce`` (SUM) at the end.

This demonstrates that not every parallel algorithm needs scatter/gather
of large data -- sometimes the parallel speedup comes purely from
replicating independent work and reducing a tiny summary at the end.

Run with:
    mpiexec -n 4 python monte_carlo_pi_parallel_mpi.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

SAMPLES_PER_RANK = 2_000_000
PI_REFERENCE = math.pi


class SequentialMonteCarloSampler:
    """Generates random samples and counts how many fall in the unit circle.

    :param samples: The number of random samples to draw.
    :param seed: The seed for this rank's independent random stream.
    """

    def __init__(self, samples: int, seed: int) -> None:
        self._samples = samples
        self._rng = random.Random(seed)

    def count_hits(self) -> int:
        """Draw :attr:`_samples` points and count how many land in the unit circle.

        :returns: The number of samples with ``x**2 + y**2 <= 1``.
        """
        hits = 0
        for _ in range(self._samples):
            x, y = self._rng.random(), self._rng.random()
            if x * x + y * y <= 1.0:
                hits += 1
        return hits


@dataclass(frozen=True)
class MonteCarloPiResult:
    """Final result of a distributed Monte Carlo Pi estimation, on rank 0.

    :param total_samples: The total number of samples drawn across all ranks.
    :param total_hits: The total number of samples landing in the unit circle.
    """

    total_samples: int
    total_hits: int

    @property
    def pi_estimate(self) -> float:
        """The estimated value of pi, ``4 * hits / samples``."""
        return 4.0 * self.total_hits / self.total_samples

    @property
    def error(self) -> float:
        """Absolute error against :data:`math.pi`."""
        return abs(self.pi_estimate - PI_REFERENCE)


class ParallelMonteCarloPiEstimator:
    """Distributed Monte Carlo Pi estimation coordinator over an MPI communicator.

    :param comm: The MPI communicator whose ranks independently sample and
        reduce their hit counts.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    @property
    def size(self) -> int:
        """The total number of ranks within :attr:`_comm`."""
        return self._size

    def run(self, samples_per_rank: int, base_seed: int) -> MonteCarloPiResult | None:
        """Estimate pi using ``samples_per_rank`` samples on every rank.

        :param samples_per_rank: The number of samples each rank draws.
        :param base_seed: A base seed; each rank adds its own rank to it
            to obtain an independent random stream.
        :returns: A :class:`MonteCarloPiResult` on rank 0, ``None`` elsewhere.
        """
        sampler = SequentialMonteCarloSampler(samples_per_rank, base_seed + self._rank)
        local_hits = sampler.count_hits()

        total_hits = self._combine_results(local_hits)

        if self._rank != 0:
            return None

        assert total_hits is not None  # guaranteed on rank 0 after `reduce(..., root=0)`
        total_samples = samples_per_rank * self._size
        return MonteCarloPiResult(total_samples=total_samples, total_hits=total_hits)

    def _combine_results(self, local_hits: int) -> int | None:
        """Sum every rank's local hit count at rank 0.

        :param local_hits: This rank's local hit count.
        :returns: The combined total on rank 0; ``None`` on every other rank.
        """
        return self._comm.reduce(local_hits, op=MPI.SUM, root=0)


def main() -> None:
    """Entry point: estimate pi using Monte Carlo sampling in parallel."""
    comm: Comm = MPI.COMM_WORLD
    estimator = ParallelMonteCarloPiEstimator(comm)

    result = estimator.run(SAMPLES_PER_RANK, base_seed=12345)

    if result is not None:
        print(f"Samples: {result.total_samples:,} across {estimator.size} ranks")
        print(f"Pi estimate: {result.pi_estimate:.6f} (error: {result.error:.6f})")


if __name__ == "__main__":
    main()
