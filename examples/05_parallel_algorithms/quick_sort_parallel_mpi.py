"""Parallel Quicksort using mpi4py (recursive process-group bisection).

Theory
------
Hyperquicksort/parallel quicksort recursively **bisects the group of
processes** along with the data, instead of bisecting only the array:

1. The current process group picks a pivot (median-of-medians via an
   ``Allgather`` of local medians) and broadcasts it.
2. Every process partitions its local data into "less" and "greater" than
   the pivot.
3. The process group splits into two halves; the lower half exchanges its
   "greater" elements with the upper half's "less" elements
   (point-to-point ``Sendrecv``), so afterward every element in the lower
   half's data is <= every element in the upper half's data.
4. Recurse independently within each half of the communicator
   (``Comm.Split``) until a group has a single process, which finishes
   with a local sequential sort.

This demonstrates **recursive communicator splitting**, a core mpi4py/MPI
technique for divide-and-conquer algorithms beyond simple scatter/gather.

Run with:
    mpiexec -n 4 python quick_sort_parallel_mpi.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm


class SequentialQuickSorter:
    """Sequential sort used as the local building block within a 1-process group.

    :param values: The list of integers to sort. Not mutated; a new sorted
        list is returned by :meth:`sort`.
    """

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def sort(self) -> list[int]:
        """Sort :attr:`_values` and return the sorted result.

        :returns: A new list containing the sorted values.
        """
        return sorted(self._values)


@dataclass(frozen=True)
class QuickSortResult:
    """Final result of a distributed hyperquicksort, available on rank 0.

    :param input_values: The original, unsorted input array.
    :param sorted_values: The fully sorted array.
    """

    input_values: list[int]
    sorted_values: list[int]

    def is_correct(self) -> bool:
        """Verify the result against Python's built-in :func:`sorted`.

        :returns: ``True`` if :attr:`sorted_values` matches
            ``sorted(input_values)``.
        """
        return self.sorted_values == sorted(self.input_values)


class ParallelHyperQuickSorter:
    """Distributed hyperquicksort coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to sort the data.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, data: list[int] | None) -> QuickSortResult | None:
        """Sort ``data`` across every rank in the communicator.

        :param data: The full, unsorted array. Only meaningful on rank 0;
            every other rank should pass ``None``.
        :returns: A :class:`QuickSortResult` on rank 0, ``None`` elsewhere.
        """
        if self._rank == 0:
            assert data is not None  # caller contract: only rank 0 passes real data
            chunks = self._split_into_chunks(data)
        else:
            chunks = None
        my_chunk = self._comm.scatter(chunks, root=0)

        sorted_local = self._hyperquicksort(self._comm, my_chunk)
        gathered = self._comm.gather(sorted_local, root=0)

        if self._rank != 0:
            return None

        assert gathered is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        result = self._combine_results(gathered)
        return QuickSortResult(input_values=data or [], sorted_values=result)

    def _split_into_chunks(self, data: list[int]) -> list[list[int]]:
        """Split ``data`` into :attr:`_size` interleaved chunks for scattering."""
        return [data[i :: self._size] for i in range(self._size)]

    def _combine_results(self, gathered: list[list[int]]) -> list[int]:
        """Flatten rank-ordered sorted chunks into the final sorted array.

        :param gathered: Per-rank sorted chunks, already globally ordered
            by recursive bisection (lower ranks hold smaller values).
        :returns: The concatenated, fully sorted array.
        """
        return [value for chunk in gathered for value in chunk]

    def _hyperquicksort(self, comm: Comm, local_data: list[int]) -> list[int]:
        """Recursively bisect ``comm`` and ``local_data`` until groups of size 1.

        :param comm: The (sub-)communicator for the current recursion level.
        :param local_data: The data local to this rank within ``comm``.
        :returns: The locally sorted data after full recursive partitioning.
        """
        size = comm.Get_size()
        rank = comm.Get_rank()

        if size == 1:
            return SequentialQuickSorter(local_data).sort()

        pivot = self._choose_pivot(comm, local_data)
        less = [x for x in local_data if x <= pivot]
        greater = [x for x in local_data if x > pivot]

        half = size // 2
        lower_group = rank < half
        partner = rank + half if lower_group else rank - half

        if partner < size:
            send_data = greater if lower_group else less
            received = comm.sendrecv(send_data, dest=partner, source=partner)
            local_data = (less if lower_group else greater) + received
        else:
            # Odd group sizes: this rank has no partner this round, keep all data.
            local_data = less + greater

        color = 0 if lower_group else 1
        sub_comm = comm.Split(color=color, key=rank)
        try:
            sorted_local = self._hyperquicksort(sub_comm, local_data)
        finally:
            sub_comm.Free()
        return sorted_local

    def _choose_pivot(self, comm: Comm, local_data: list[int]) -> float:
        """Pick a pivot as the average of local medians across ``comm``.

        :param comm: The (sub-)communicator for the current recursion level.
        :param local_data: The data local to this rank within ``comm``.
        :returns: A pivot value used to bisect every rank's local data.
        """
        local_median = sorted(local_data)[len(local_data) // 2] if local_data else 0
        all_medians: list[float] = comm.allgather(local_median)
        return sum(all_medians) / len(all_medians)


def main() -> None:
    """Entry point: generate random data on rank 0 and sort it in parallel."""
    comm: Comm = MPI.COMM_WORLD
    sorter = ParallelHyperQuickSorter(comm)

    data: list[int] | None = None
    if sorter.rank == 0:
        random.seed(7)
        data = [random.randint(0, 500) for _ in range(32)]

    result = sorter.run(data)

    if result is not None:
        print(f"Input : {result.input_values}")
        print(f"Sorted: {result.sorted_values}")
        assert result.is_correct(), "Parallel hyperquicksort failed!"


if __name__ == "__main__":
    main()
