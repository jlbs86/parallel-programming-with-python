"""Parallel Depth-First Search (DFS) using mpi4py (task decomposition model).

Theory
------
DFS is inherently sequential along a single path (you must finish exploring
one branch before backtracking), so naive parallelization at the vertex
level gives poor speedup. The standard approach -- used here -- is to
parallelize at the level of *independent subtrees*: once the search tree
branches, each branch can be explored by a different process.

This example implements a **master-worker (task-parallel) decomposition**:

1. Rank 0 performs a shallow DFS until the frontier has at least ``size``
   independent branches (or the graph is exhausted).
2. Each branch is distributed to a worker rank with ``scatter``.
3. Each worker runs a full sequential DFS rooted at its assigned vertex,
   restricted to the globally-unvisited set it was given.
4. Results (visited sets and discovery order) are combined with ``gather``.

This mirrors how real work-stealing / fork-join DFS frameworks (e.g.
parallel SAT solvers, parallel branch-and-bound) split exploration once
enough parallel work exists, instead of trying to parallelize a single
sequential path.

.. note::
   This simplified example assigns at most one branch per rank and does
   not re-queue leftover branches onto workers that finish early. It
   fully visits the graph when ``size`` is at least the number of
   branches produced by :meth:`ParallelDepthFirstSearchCoordinator._split_into_branches`
   (true for ``size >= 2`` on :data:`GRAPH`). A production work-stealing
   scheduler would keep a shared task queue and re-dispatch leftover
   branches to idle workers instead of dropping them, which is what
   makes ``size == 1`` incomplete here.

Run with:
    mpiexec -n 4 python dfs_parallel_mpi.py
"""

from __future__ import annotations

from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

GRAPH: dict[int, list[int]] = {
    0: [1, 2, 3],
    1: [4],
    2: [5],
    3: [6],
    4: [],
    5: [7],
    6: [],
    7: [],
}
SOURCE = 0


class SequentialDepthFirstSearcher:
    """Iterative DFS restricted to vertices not already visited.

    :param graph: Adjacency list mapping each vertex to its neighbors.
    :param visited: The set of already-visited vertices, shared with the
        caller; mutated in place as new vertices are discovered.
    """

    def __init__(self, graph: dict[int, list[int]], visited: set[int]) -> None:
        self._graph = graph
        self._visited = visited

    def run(self, start: int) -> list[int]:
        """Run DFS from ``start`` and return the discovery order.

        :param start: The vertex to begin the search from.
        :returns: Vertices in the order they were first visited.
        """
        order: list[int] = []
        stack = [start]
        while stack:
            vertex = stack.pop()
            if vertex in self._visited:
                continue
            self._visited.add(vertex)
            order.append(vertex)
            # Reverse to preserve the natural left-to-right visiting order.
            for neighbor in reversed(self._graph.get(vertex, [])):
                if neighbor not in self._visited:
                    stack.append(neighbor)
        return order


@dataclass(frozen=True)
class DepthFirstSearchResult:
    """Final result of a distributed, task-parallel DFS, available on rank 0.

    :param source: The vertex the search started from.
    :param visit_order: The full combined visit order across all branches.
    """

    source: int
    visit_order: list[int]


class ParallelDepthFirstSearchCoordinator:
    """Master-worker DFS coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to explore
        independent subtrees of the graph.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, graph: dict[int, list[int]], source: int) -> DepthFirstSearchResult | None:
        """Explore ``graph`` from ``source`` using independent-subtree tasking.

        :param graph: Adjacency list for the graph. Only meaningful on rank
            0 in this example; every rank holds the same constant graph.
        :param source: The root vertex to search from.
        :returns: A :class:`DepthFirstSearchResult` on rank 0, ``None``
            elsewhere.
        """
        if self._rank == 0:
            branch_roots, master_order = self._split_into_branches(graph, source, self._size)
            tasks: list[int | None] = list(branch_roots[: self._size])
            while len(tasks) < self._size:
                tasks.append(None)
            visited_by_master: set[int] | None = set(master_order)
        else:
            tasks = []
            master_order = None
            visited_by_master = None

        my_task = self._comm.scatter(tasks, root=0)
        visited_by_master = self._comm.bcast(visited_by_master, root=0)

        local_order = self._explore_branch(graph, my_task, visited_by_master)
        all_orders = self._comm.gather(local_order, root=0)

        if self._rank != 0:
            return None

        assert master_order is not None  # guaranteed on rank 0, computed above
        assert all_orders is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        return self._combine_results(source, master_order, all_orders)

    def _split_into_branches(
        self, graph: dict[int, list[int]], source: int, needed: int
    ) -> tuple[list[int], list[int]]:
        """Expand ``source`` breadth-first just enough to produce ``needed`` roots.

        Vertices are only marked visited once we commit to expanding
        *past* them (i.e. once we know their children, not themselves,
        will become the next candidate branch roots). The vertices in the
        frontier returned to the caller are always left out of
        ``master_order`` so that the worker assigned to explore one does
        not immediately treat it as already seen.

        :param graph: Adjacency list for the graph.
        :param source: The root vertex to expand from.
        :param needed: The minimum number of branch roots desired.
        :returns: A tuple of ``(branch_roots, master_order)``, where
            ``master_order`` lists ``source`` and every intermediate
            vertex the master itself discovered while searching for
            enough branch roots, in discovery order.
        """
        master_order = [source]
        visited = {source}
        frontier = list(graph.get(source, []))

        while len(frontier) < needed:
            expandable = [v for v in frontier if graph.get(v)]
            if not expandable:
                break  # No frontier vertex has children left to reveal.

            next_frontier = [v for v in frontier if v not in expandable]
            for vertex in expandable:
                visited.add(vertex)  # Committing to expand past `vertex`.
                master_order.append(vertex)
                for neighbor in graph.get(vertex, []):
                    if neighbor not in visited and neighbor not in next_frontier:
                        next_frontier.append(neighbor)

            frontier = next_frontier

        return frontier, master_order

    def _explore_branch(
        self, graph: dict[int, list[int]], task: int | None, visited_by_master: set[int]
    ) -> list[int]:
        """Run a local sequential DFS on this rank's assigned branch, if any.

        :param graph: Adjacency list for the graph.
        :param task: The branch root assigned to this rank, or ``None`` if idle.
        :param visited_by_master: The globally-known visited set to avoid
            re-exploring vertices already claimed by the master's split.
        :returns: The discovery order produced by this rank, or an empty
            list if this rank was idle.
        """
        if task is None:
            return []
        local_visited = set(visited_by_master)
        return SequentialDepthFirstSearcher(graph, local_visited).run(task)

    def _combine_results(
        self, source: int, master_order: list[int], all_orders: list[list[int]]
    ) -> DepthFirstSearchResult:
        """Combine the master's and every worker's branch order into one result.

        :param source: The root vertex the search started from.
        :param master_order: Vertices discovered by rank 0 itself while
            splitting the graph into branches (includes ``source``).
        :param all_orders: Per-rank discovery orders, in rank order.
        :returns: The combined :class:`DepthFirstSearchResult`.
        """
        full_order = master_order + [vertex for branch in all_orders for vertex in branch]
        return DepthFirstSearchResult(source=source, visit_order=full_order)


def main() -> None:
    """Entry point: explore :data:`GRAPH` from :data:`SOURCE` in parallel."""
    comm: Comm = MPI.COMM_WORLD
    coordinator = ParallelDepthFirstSearchCoordinator(comm)

    result = coordinator.run(GRAPH, SOURCE)

    if result is not None:
        print(f"DFS visit order (task-parallel) from vertex {result.source}: {result.visit_order}")


if __name__ == "__main__":
    main()
