"""Parallel Odd-Even Transposition Sort using mpi4py.

Theory
------
Bubble sort is O(n^2) and famously sequential: each pass depends on the
result of the previous one. Its well known parallel analogue is
**odd-even transposition sort**, a sorting network that performs the same
"compare-and-swap-adjacent" idea but applies *all* independent swaps of a
phase simultaneously:

- Odd phase: compare-swap pairs (1,2), (3,4), (5,6), ...
- Even phase: compare-swap pairs (0,1), (2,3), (4,5), ...

With ``n`` elements split one-per-process across ``n`` ranks, each phase
is a single round of point-to-point communication (``Sendrecv``) between
neighboring ranks, and the whole sort finishes in exactly ``n`` phases --
this is a textbook example of a **sorting network** mapped directly onto
distributed-memory message passing.

Run with:
    mpiexec -n 8 python bubble_sort_parallel_mpi.py
"""

from __future__ import annotations

from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

DEFAULT_VALUES: list[int] = [37, 2, 19, 4, 25, 1, 33, 8]


class SequentialOddEvenSorter:
    """Local single-value holder used as the per-rank unit of a sorting network.

    :param value: The single integer owned by this rank.
    """

    def __init__(self, value: int) -> None:
        self._value = value

    def compare_and_keep_min(self, other: int) -> int:
        """Return the smaller of :attr:`_value` and ``other``.

        :param other: The value received from a neighboring rank.
        :returns: ``min(self._value, other)``.
        """
        self._value = min(self._value, other)
        return self._value

    def compare_and_keep_max(self, other: int) -> int:
        """Return the larger of :attr:`_value` and ``other``.

        :param other: The value received from a neighboring rank.
        :returns: ``max(self._value, other)``.
        """
        self._value = max(self._value, other)
        return self._value

    @property
    def value(self) -> int:
        """The current value held by this rank."""
        return self._value


@dataclass(frozen=True)
class OddEvenSortResult:
    """Final result of a distributed odd-even transposition sort, on rank 0.

    :param input_values: The original, unsorted per-rank values.
    :param sorted_values: The fully sorted values, in rank order.
    """

    input_values: list[int]
    sorted_values: list[int]

    def is_correct(self) -> bool:
        """Verify the result against Python's built-in :func:`sorted`.

        :returns: ``True`` if :attr:`sorted_values` matches
            ``sorted(input_values)``.
        """
        return self.sorted_values == sorted(self.input_values)


class ParallelOddEvenSorter:
    """Distributed odd-even transposition sort coordinator over an MPI communicator.

    :param comm: The MPI communicator; each rank owns exactly one element.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, unsorted_values: list[int] | None) -> OddEvenSortResult | None:
        """Run the full odd-even transposition sort across the communicator.

        :param unsorted_values: The complete list of per-rank values, with
            ``unsorted_values[r]`` owned by rank ``r``. Only meaningful on
            rank 0; every other rank should pass ``None``.
        :returns: An :class:`OddEvenSortResult` on rank 0, ``None`` elsewhere.
        """
        unsorted_values = self._comm.bcast(unsorted_values, root=0)
        local_sorter = SequentialOddEvenSorter(unsorted_values[self._rank])

        for phase in range(self._size):
            self._run_phase(phase, local_sorter)

        sorted_values = self._comm.gather(local_sorter.value, root=0)

        if self._rank != 0:
            return None

        assert sorted_values is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        return OddEvenSortResult(input_values=unsorted_values, sorted_values=sorted_values)

    def _partner_for_phase(self, phase: int) -> int:
        """Compute the neighboring rank this rank exchanges with in ``phase``.

        :param phase: The current phase index (0-based).
        :returns: The partner rank, which may be out of range at the edges.
        """
        if phase % 2 == 0:
            return self._rank + 1 if self._rank % 2 == 0 else self._rank - 1
        return self._rank - 1 if self._rank % 2 == 0 else self._rank + 1

    def _run_phase(self, phase: int, local_sorter: SequentialOddEvenSorter) -> None:
        """Exchange and compare-swap with the partner rank for one phase.

        :param phase: The current phase index (0-based).
        :param local_sorter: The local single-value sorter to update in place.
        """
        partner = self._partner_for_phase(phase)

        if 0 <= partner < self._size:
            received = self._comm.sendrecv(local_sorter.value, dest=partner, source=partner)
            if self._rank < partner:
                local_sorter.compare_and_keep_min(received)
            else:
                local_sorter.compare_and_keep_max(received)

        self._comm.Barrier()


def main() -> None:
    """Entry point: seed per-rank values on rank 0 and sort them in parallel."""
    comm: Comm = MPI.COMM_WORLD
    sorter = ParallelOddEvenSorter(comm)

    unsorted_values: list[int] | None = None
    if sorter.rank == 0:
        size = comm.Get_size()
        unsorted_values = (
            DEFAULT_VALUES[:size] if size <= len(DEFAULT_VALUES) else list(range(size, 0, -1))
        )

    result = sorter.run(unsorted_values)

    if result is not None:
        print(f"Input : {result.input_values}")
        print(f"Sorted: {result.sorted_values}")
        assert result.is_correct(), "Odd-even transposition sort failed!"


if __name__ == "__main__":
    main()
