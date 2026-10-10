"""Parallel Search (linear and binary) using mpi4py (data decomposition).

Theory
------
Searching an unsorted collection for a target value is embarrassingly
parallel: splitting the data across processes and searching each slice
independently is correct because "does X exist anywhere" is simply the
logical OR of "does X exist in my slice" over every process.

Searching a *sorted* distributed array for the first index >= target is a
different, more interesting problem: each process holds a contiguous
sorted block, and at most one block can contain the answer. We solve it in
two parallel phases:

1. **Parallel linear search** (unsorted case): every rank scans its local
   chunk; results are combined with ``allreduce`` (logical OR) for a
   found/not-found answer and ``gather`` to collect every matching
   position.
2. **Parallel binary search over sorted, block-distributed data**: every
   rank knows the first and last element of every other rank's block (one
   cheap ``allgather`` of block boundaries), so it can determine locally,
   without further communication, whether the target could be in its
   block, then runs a local sequential binary search only if so.

Run with:
    mpiexec -n 4 python parallel_search_mpi.py
"""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

TARGET = 777


class SequentialLinearSearcher:
    """Scans a local chunk of an unsorted array for a target value.

    :param local_data: The chunk of the unsorted array owned by this rank.
    """

    def __init__(self, local_data: list[int]) -> None:
        self._local_data = local_data

    def find_local_matches(self, target: int, offset: int) -> list[int]:
        """Find every local index matching ``target``, reported as global indices.

        :param target: The value to search for.
        :param offset: This rank's starting offset within the original array.
        :returns: Global indices (within the original array) matching ``target``.
        """
        return [offset + i for i, value in enumerate(self._local_data) if value == target]


class SequentialBinarySearcher:
    """Runs a local binary search within a sorted block.

    :param local_sorted_block: The sorted block of values owned by this rank.
    """

    def __init__(self, local_sorted_block: list[int]) -> None:
        self._local_sorted_block = local_sorted_block

    def find(self, target: int) -> int | None:
        """Binary-search :attr:`_local_sorted_block` for ``target``.

        :param target: The value to search for.
        :returns: The local index of ``target`` within the block, or
            ``None`` if absent.
        """
        pos = bisect.bisect_left(self._local_sorted_block, target)
        if pos < len(self._local_sorted_block) and self._local_sorted_block[pos] == target:
            return pos
        return None

    @property
    def bounds(self) -> tuple[int, int] | None:
        """The ``(first, last)`` values of the block, or ``None`` if empty."""
        if not self._local_sorted_block:
            return None
        return self._local_sorted_block[0], self._local_sorted_block[-1]


@dataclass(frozen=True)
class ParallelSearchResult:
    """Final result of both parallel search phases, available on rank 0.

    :param target: The value that was searched for.
    :param linear_matches: Global indices found via parallel linear search.
    :param binary_match: Global index found via parallel binary search, or
        ``None`` if absent.
    """

    target: int
    linear_matches: list[int]
    binary_match: int | None


class ParallelSearchCoordinator:
    """Distributed search coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to run both the
        linear and binary search phases.
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
        self,
        local_unsorted_chunk: list[int],
        local_sorted_block: list[int],
        target: int,
    ) -> ParallelSearchResult | None:
        """Run both parallel search phases over data already scattered to ranks.

        :param local_unsorted_chunk: This rank's chunk of the unsorted array.
        :param local_sorted_block: This rank's contiguous block of the
            sorted array.
        :param target: The value to search for.
        :returns: A :class:`ParallelSearchResult` on rank 0, ``None`` elsewhere.
        """
        linear_matches = self._parallel_linear_search(local_unsorted_chunk, target)
        binary_match = self._parallel_binary_search(local_sorted_block, target)

        if self._rank != 0:
            return None

        return ParallelSearchResult(
            target=target, linear_matches=linear_matches, binary_match=binary_match
        )

    def _parallel_linear_search(self, local_data: list[int], target: int) -> list[int]:
        """Search ``local_data`` across every rank and combine the matches.

        :param local_data: This rank's chunk of the unsorted array.
        :param target: The value to search for.
        :returns: The global indices matching ``target``, broadcast to
            every rank (meaningful return value on all ranks).
        """
        counts = self._comm.allgather(len(local_data))
        offset = sum(counts[: self._rank])

        local_matches = SequentialLinearSearcher(local_data).find_local_matches(target, offset)
        found_anywhere = self._comm.allreduce(bool(local_matches), op=MPI.LOR)

        if not found_anywhere:
            return []

        gathered = self._comm.gather(local_matches, root=0)
        if self._rank == 0:
            assert gathered is not None  # guaranteed on rank 0 after `gather(..., root=0)`
            flattened: list[int] | None = [idx for group in gathered for idx in group]
        else:
            flattened = None
        all_matches: list[int] | None = self._comm.bcast(flattened, root=0)
        return all_matches if all_matches is not None else []

    def _parallel_binary_search(self, local_sorted_block: list[int], target: int) -> int | None:
        """Binary-search a sorted, block-distributed array for ``target``.

        :param local_sorted_block: This rank's contiguous block of the
            sorted array.
        :param target: The value to search for.
        :returns: The global index of ``target``, or ``None`` if absent
            (meaningful return value on all ranks).
        """
        searcher = SequentialBinarySearcher(local_sorted_block)
        all_bounds = self._comm.allgather(searcher.bounds)

        # Collective call: every rank must participate unconditionally, so
        # this happens before the per-rank branch below (not inside it).
        counts = self._comm.allgather(len(local_sorted_block))

        candidate_rank = self._find_candidate_rank(all_bounds, target)
        if candidate_rank is None:
            return None

        local_result = None
        if self._rank == candidate_rank:
            pos = searcher.find(target)
            if pos is not None:
                offset = sum(counts[: self._rank])
                local_result = offset + pos

        result: int | None = self._comm.bcast(local_result, root=candidate_rank)
        return result

    def _find_candidate_rank(
        self, all_bounds: list[tuple[int, int] | None], target: int
    ) -> int | None:
        """Determine which rank's block could contain ``target``.

        :param all_bounds: Every rank's ``(first, last)`` block bounds.
        :param target: The value to search for.
        :returns: The rank whose block bounds include ``target``, or
            ``None`` if no block can contain it.
        """
        for r, bounds in enumerate(all_bounds):
            if bounds is not None and bounds[0] <= target <= bounds[1]:
                return r
        return None


def main() -> None:
    """Entry point: build demo data on rank 0 and run both search phases."""
    comm: Comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        random.seed(5)
        unsorted_data = [random.randint(0, 1000) for _ in range(40)]
        unsorted_data[13] = TARGET  # guarantee a hit for the demo
        chunks_unsorted = [unsorted_data[i::size] for i in range(size)]

        sorted_data = sorted(random.randint(0, 1000) for _ in range(40))
        sorted_data[20] = TARGET
        sorted_data.sort()
        block_size = len(sorted_data) // size
        chunks_sorted = [
            sorted_data[i * block_size : None if i == size - 1 else (i + 1) * block_size]
            for i in range(size)
        ]
    else:
        chunks_unsorted = None
        chunks_sorted = None

    my_unsorted_chunk = comm.scatter(chunks_unsorted, root=0)
    my_sorted_block = comm.scatter(chunks_sorted, root=0)

    coordinator = ParallelSearchCoordinator(comm)
    result = coordinator.run(my_unsorted_chunk, my_sorted_block, TARGET)

    if result is not None:
        print(
            f"[Linear search] target {result.target} found at global indices: "
            f"{result.linear_matches}"
        )
        print(
            f"[Binary search]  target {result.target} found at global index: "
            f"{result.binary_match}"
        )


if __name__ == "__main__":
    main()
