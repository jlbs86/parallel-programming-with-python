"""Parallel Reduce and Scan (prefix sum) using mpi4py.

Theory
------
Reduce and scan are the two foundational collective operations behind
almost every other parallel algorithm in this repository:

- **Reduce**: combine a value from every process into a single result on
  one process (or every process, with ``Allreduce``) using an associative
  operator (sum, max, min, logical and/or, ...). Implemented internally as
  a *reduction tree* (O(log P) communication steps instead of O(P)).
- **Scan** (parallel prefix sum): compute, for every process ``i``, the
  reduction of all values from processes ``0..i`` (inclusive scan). This
  is the parallel generalization of a running total and is the building
  block behind parallel sorting, stream compaction, and load balancing.

mpi4py exposes both directly as thin wrappers over MPI's native
``MPI_Reduce`` / ``MPI_Allreduce`` / ``MPI_Scan`` / ``MPI_Exscan``, which
use optimized tree or butterfly algorithms under the hood -- this example
also implements a **manual recursive-doubling scan** with point-to-point
messages so the O(log P) mechanics are visible instead of hidden behind a
single collective call.

Run with:
    mpiexec -n 8 python parallel_reduce_scan_mpi.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from mpi4py import MPI
from mpi4py.MPI import Comm, Intracomm


class SequentialRecursiveDoublingScanner:
    """Performs a manual recursive-doubling inclusive prefix sum.

    Classic Hillis-Steele scan: at each round ``d``, every rank whose
    index is at least ``2**d`` receives the running total from the rank
    ``2**d`` positions below it and adds it in; ranks below that offset
    keep their value unchanged for this round. After ``ceil(log2(size))``
    rounds every rank holds the inclusive prefix sum of all ranks up to
    itself.

    :param comm: The MPI communicator to scan over.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    def scan(self, value: int) -> int:
        """Compute the inclusive prefix sum up to and including this rank.

        :param value: This rank's local value to include in the scan.
        :returns: The inclusive prefix sum through this rank.
        """
        total = value
        step = 1
        while step < self._size:
            if self._rank - step >= 0:
                dest = self._rank + step if self._rank + step < self._size else MPI.PROC_NULL
                received = self._comm.sendrecv(total, dest=dest, source=self._rank - step)
                total += received
            elif self._rank + step < self._size:
                self._comm.send(total, dest=self._rank + step)
            step *= 2
        return total


@dataclass(frozen=True)
class ReduceScanResult:
    """Per-rank result of the reduce and scan demonstration.

    :param rank: The rank this result belongs to.
    :param value: This rank's original local value.
    :param inclusive_scan_builtin: Inclusive prefix sum from ``MPI_Scan``.
    :param inclusive_scan_manual: Inclusive prefix sum from the manual
        recursive-doubling implementation.
    :param exclusive_scan: Exclusive prefix sum from ``MPI_Exscan``
        (undefined on rank 0).
    :param total_sum: Reduced sum across all ranks, if computed on this rank.
    :param global_max: Reduced max across all ranks, if computed on this rank.
    :param total_sum_everywhere: Allreduced sum, available on every rank.
    """

    rank: int
    value: int
    inclusive_scan_builtin: int
    inclusive_scan_manual: int
    exclusive_scan: int | None
    total_sum: int | None
    global_max: int | None
    total_sum_everywhere: int

    def scans_agree(self) -> bool:
        """Check that the manual scan matches the built-in ``MPI_Scan``.

        :returns: ``True`` if both inclusive scans agree.
        """
        return self.inclusive_scan_builtin == self.inclusive_scan_manual


class ParallelReduceScanCoordinator:
    """Reduce/scan demonstration coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to compute
        reductions and prefix sums.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, value: int) -> ReduceScanResult:
        """Run every reduce/scan variant for ``value`` on this rank.

        :param value: This rank's local value.
        :returns: A :class:`ReduceScanResult` describing this rank's view.
        """
        total_sum = self._comm.reduce(value, op=MPI.SUM, root=0)
        global_max = self._comm.reduce(value, op=MPI.MAX, root=0)
        total_sum_everywhere = self._comm.allreduce(value, op=MPI.SUM)
        intracomm = cast(Intracomm, self._comm)
        builtin_prefix_sum = intracomm.scan(value, op=MPI.SUM)
        exclusive_prefix_sum = intracomm.exscan(value, op=MPI.SUM)

        manual_prefix_sum = SequentialRecursiveDoublingScanner(self._comm).scan(value)

        return ReduceScanResult(
            rank=self._rank,
            value=value,
            inclusive_scan_builtin=builtin_prefix_sum,
            inclusive_scan_manual=manual_prefix_sum,
            exclusive_scan=exclusive_prefix_sum,
            total_sum=total_sum,
            global_max=global_max,
            total_sum_everywhere=total_sum_everywhere,
        )


def main() -> None:
    """Entry point: run reduce/scan demonstrations across every rank."""
    comm: Comm = MPI.COMM_WORLD
    coordinator = ParallelReduceScanCoordinator(comm)

    value = coordinator.rank + 1  # ranks hold 1, 2, 3, ..., size
    result = coordinator.run(value)

    print(
        f"[rank {result.rank}] value={result.value} "
        f"inclusive_scan(builtin)={result.inclusive_scan_builtin} "
        f"inclusive_scan(manual)={result.inclusive_scan_manual} "
        f"exclusive_scan={result.exclusive_scan}"
    )

    if coordinator.rank == 0:
        print(f"\nTotal sum (reduce, root only): {result.total_sum}")
        print(f"Max value (reduce, root only): {result.global_max}")
        print(f"Total sum (allreduce, every rank): {result.total_sum_everywhere}")
        assert result.scans_agree(), "Manual scan disagrees with MPI_Scan!"


if __name__ == "__main__":
    main()
