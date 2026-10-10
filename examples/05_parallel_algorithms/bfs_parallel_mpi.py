"""Parallel Breadth-First Search (BFS) using mpi4py (distributed memory model).

Theory
------
BFS explores a graph level by level. The parallel "level-synchronous" BFS
decomposes the *frontier* (the set of vertices discovered at the current
level) across processes instead of decomposing the graph itself. Each
process:

1. Receives the current frontier (broadcast from rank 0).
2. Expands only the slice of the frontier assigned to it, producing local
   candidate neighbors.
3. Participates in an ``Allgather`` so every process ends the level with the
   full next frontier and the full set of globally visited vertices.

This is a classic example of **data decomposition + collective
communication** (Broadcast + Allgather) rather than task decomposition, and
it mirrors how distributed graph engines (Pregel, GraphX) implement
level-synchronous BFS at scale.

Run with:
    mpiexec -n 4 python bfs_parallel_mpi.py
"""

from __future__ import annotations

from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

# A toy adjacency list. In production this would be partitioned across
# nodes (e.g. via METIS); here every rank holds the full graph for clarity
# and only the *frontier* is parallelized.
GRAPH: dict[int, list[int]] = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1],
    4: [1, 5],
    5: [2, 4, 6],
    6: [5],
}
SOURCE = 0


class SequentialFrontierExpander:
    """Expands a slice of the BFS frontier against a fixed graph.

    :param graph: Adjacency list mapping each vertex to its neighbors.
    """

    def __init__(self, graph: dict[int, list[int]]) -> None:
        self._graph = graph

    def expand(self, frontier_slice: list[int], visited: set[int]) -> set[int]:
        """Return every unvisited neighbor reachable from ``frontier_slice``.

        :param frontier_slice: The subset of the current frontier owned by
            the calling rank.
        :param visited: The globally-known visited set, used read-only here.
        :returns: The set of newly discovered neighbor vertices.
        """
        local_next: set[int] = set()
        for vertex in frontier_slice:
            for neighbor in self._graph.get(vertex, []):
                if neighbor not in visited:
                    local_next.add(neighbor)
        return local_next


@dataclass(frozen=True)
class BreadthFirstSearchResult:
    """Final result of a distributed, level-synchronous BFS.

    :param source: The vertex the search started from.
    :param visit_order: Vertices in the order they were first visited.
    """

    source: int
    visit_order: list[int]


class ParallelBreadthFirstSearchCoordinator:
    """Level-synchronous BFS coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to expand each
        frontier level in parallel.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, graph: dict[int, list[int]], source: int) -> BreadthFirstSearchResult:
        """Explore ``graph`` from ``source`` level by level, in parallel.

        :param graph: Adjacency list for the graph. Every rank holds the
            same full graph in this example.
        :param source: The root vertex to search from.
        :returns: The :class:`BreadthFirstSearchResult`, identical on every
            rank since every rank ends each level with the full state.
        """
        expander = SequentialFrontierExpander(graph)

        visited = {source}
        order = [source]
        frontier = [source]

        while frontier:
            my_slice = self._local_slice(frontier)
            local_next = expander.expand(my_slice, visited)
            next_frontier = self._combine_frontier(local_next, visited, order)
            frontier = next_frontier

        return BreadthFirstSearchResult(source=source, visit_order=order)

    def _local_slice(self, items: list[int]) -> list[int]:
        """Return the contiguous chunk of ``items`` owned by this rank.

        :param items: The full list to be sliced across ranks.
        :returns: The slice of ``items`` assigned to this rank.
        """
        n = len(items)
        chunk, remainder = divmod(n, self._size)
        start = self._rank * chunk + min(self._rank, remainder)
        end = start + chunk + (1 if self._rank < remainder else 0)
        return items[start:end]

    def _combine_frontier(
        self, local_next: set[int], visited: set[int], order: list[int]
    ) -> list[int]:
        """Combine every rank's discoveries into the next frontier.

        :param local_next: This rank's locally discovered candidate vertices.
        :param visited: The globally-known visited set; updated in place.
        :param order: The accumulated discovery order; updated in place.
        :returns: The next frontier to expand, in deterministic order.
        """
        all_next: list[set[int]] = self._comm.allgather(local_next)

        next_frontier: list[int] = []
        for candidates in all_next:
            for vertex in sorted(candidates):
                if vertex not in visited:
                    visited.add(vertex)
                    next_frontier.append(vertex)
                    order.append(vertex)

        return next_frontier


def main() -> None:
    """Entry point: explore :data:`GRAPH` from :data:`SOURCE` in parallel."""
    comm: Comm = MPI.COMM_WORLD
    coordinator = ParallelBreadthFirstSearchCoordinator(comm)

    result = coordinator.run(GRAPH, SOURCE)

    if coordinator.rank == 0:
        print(f"BFS visit order from vertex {result.source}: {result.visit_order}")


if __name__ == "__main__":
    main()
