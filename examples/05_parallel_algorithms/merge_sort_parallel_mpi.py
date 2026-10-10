"""Parallel Merge Sort using mpi4py (divide-and-conquer, data decomposition).

Theory
------
Merge sort is naturally parallel: splitting the array is independent work,
and only the merge step requires combining results. This is a canonical
example of the **divide-and-conquer parallel pattern**:

1. Rank 0 splits the array into ``size`` contiguous chunks (``scatter``).
2. Every rank sorts its own chunk independently and in parallel -- no
   communication needed during this phase (embarrassingly parallel).
3. Sorted chunks are sent back to rank 0 (``gather``) and merged pairwise,
   which is the classic **binary merge tree** reduction, here done with
   :func:`heapq.merge` for an O(n log k) k-way merge instead of repeated
   pairwise merges.

Run with::

    mpiexec -n 4 python merge_sort_parallel_mpi.py
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm


class SequentialMergeSorter:
    """Sequential merge sort used as the local building block on each rank.

    :param values: The list of integers to sort. Not mutated; a new sorted
        list is returned by :meth:`sort`.
    """

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def sort(self) -> list[int]:
        """Recursively sort :attr:`_values` and return the sorted result.

        :returns: A new list containing the sorted values.
        """
        return self._merge_sort(self._values)

    def _merge_sort(self, values: list[int]) -> list[int]:
        if len(values) <= 1:
            return values
        mid = len(values) // 2
        left = self._merge_sort(values[:mid])
        right = self._merge_sort(values[mid:])
        return list(heapq.merge(left, right))


@dataclass(frozen=True)
class MergeSortResult:
    """Final result of a distributed merge sort, available on rank 0.

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


class ParallelMergeSorter:
    """Distributed merge sort coordinator built on top of an MPI communicator.

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

    def run(self, data: list[int] | None) -> MergeSortResult | None:
        """Sort ``data`` across every rank in the communicator.

        :param data: The full, unsorted array. Only meaningful on rank 0;
            every other rank should pass ``None``.
        :returns: A :class:`MergeSortResult` on rank 0, ``None`` elsewhere.
        """
        if self._rank == 0:
            assert data is not None  # caller contract: only rank 0 passes real data
            chunks = self._split_into_chunks(data)
        else:
            chunks = None
        local_chunk = self._comm.scatter(chunks, root=0)

        locally_sorted = SequentialMergeSorter(local_chunk).sort()
        gathered_chunks = self._comm.gather(locally_sorted, root=0)

        if self._rank != 0:
            return None

        assert gathered_chunks is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        merged = list(heapq.merge(*gathered_chunks))
        return MergeSortResult(input_values=data or [], sorted_values=merged)

    def _split_into_chunks(self, data: list[int]) -> list[list[int]]:
        """Split ``data`` into :attr:`_size` interleaved chunks for scattering."""
        return [data[i :: self._size] for i in range(self._size)]


def main() -> None:
    """Entry point: generate random data on rank 0 and sort it in parallel."""
    comm: Comm = MPI.COMM_WORLD
    sorter = ParallelMergeSorter(comm)

    data: list[int] | None = None
    if sorter.rank == 0:
        random.seed(42)
        data = [random.randint(0, 1000) for _ in range(40)]

    result = sorter.run(data)

    if result is not None:
        print(f"Input : {result.input_values}")
        print(f"Sorted: {result.sorted_values}")
        assert result.is_correct(), "Parallel merge sort failed!"


if __name__ == "__main__":
    main()
