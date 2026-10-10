"""Parallel Dijkstra's shortest path using mpi4py (delta-stepping-inspired).

Theory
------
Classic Dijkstra is sequential: it repeatedly extracts the global minimum
from a priority queue. The extraction step is a hard synchronization point,
so naive parallelization is dominated by communication.

This example implements the well known **parallel relaxation** scheme used
by distributed shortest-path solvers:

1. The vertex set is partitioned across processes (data decomposition);
   each rank owns a contiguous slice of vertices and their outgoing edges.
2. In each round, every process relaxes edges out of vertices that were
   updated in the previous round *and that it owns*.
3. Updated distances to vertices owned by other ranks are exchanged with an
   ``Allreduce`` (element-wise MIN) over the full distance vector.
4. Rounds repeat until no distance changes anywhere (global convergence
   checked via ``Allreduce`` with logical OR).

This is the same "relax until convergence" idea behind Bellman-Ford and
delta-stepping, adapted so it parallelizes cleanly with collective
operations instead of a centralized priority queue.

Run with:
    mpiexec -n 4 python dijkstra_parallel_mpi.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm

INF = math.inf

# Weighted directed graph: vertex -> list of (neighbor, weight)
GRAPH: dict[int, list[tuple[int, float]]] = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: [(4, 3)],
    4: [],
}
SOURCE = 0
NUM_VERTICES = len(GRAPH)


class SequentialVertexOwnership:
    """Static block-ownership mapping from vertex id to owning rank.

    :param num_vertices: The total number of vertices in the graph.
    :param size: The number of MPI ranks participating.
    """

    def __init__(self, num_vertices: int, size: int) -> None:
        self._num_vertices = num_vertices
        self._size = size
        self._chunk = math.ceil(num_vertices / size)

    def owner_of(self, vertex: int) -> int:
        """Return the rank that owns ``vertex``.

        :param vertex: The vertex id to look up.
        :returns: The owning rank.
        """
        return min(vertex // self._chunk, self._size - 1)

    def vertices_owned_by(self, rank: int, graph: dict[int, list[tuple[int, float]]]) -> list[int]:
        """Return every vertex in ``graph`` owned by ``rank``.

        :param rank: The rank to collect vertices for.
        :param graph: The adjacency list used to enumerate vertex ids.
        :returns: The list of vertices owned by ``rank``.
        """
        return [v for v in graph if self.owner_of(v) == rank]


@dataclass(frozen=True)
class ShortestPathResult:
    """Final result of a distributed Dijkstra relaxation, available on rank 0.

    :param source: The source vertex distances are measured from.
    :param distances: Shortest distance from :attr:`source` to each vertex,
        indexed by vertex id.
    :param rounds: The number of synchronization rounds needed to converge.
    """

    source: int
    distances: list[float]
    rounds: int


class ParallelDijkstraSolver:
    """Distributed shortest-path coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to relax
        distances until convergence.
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
        self, graph: dict[int, list[tuple[int, float]]], source: int, num_vertices: int
    ) -> ShortestPathResult | None:
        """Compute shortest distances from ``source`` across every vertex.

        :param graph: Adjacency list; every rank holds the full graph in
            this example, but only relaxes the vertices it owns.
        :param source: The vertex to compute distances from.
        :param num_vertices: The total number of vertices in ``graph``.
        :returns: A :class:`ShortestPathResult` on rank 0, ``None`` elsewhere.
        """
        ownership = SequentialVertexOwnership(num_vertices, self._size)
        my_vertices = ownership.vertices_owned_by(self._rank, graph)

        dist = [INF] * num_vertices
        dist[source] = 0.0

        changed = True
        rounds = 0
        while changed:
            rounds += 1
            local_updates = self._relax_owned_vertices(graph, my_vertices, dist)
            new_dist = self._combine_distances(local_updates)
            changed = self._has_converged(new_dist, dist)
            dist = new_dist

        if self._rank != 0:
            return None

        return ShortestPathResult(source=source, distances=dist, rounds=rounds)

    def _relax_owned_vertices(
        self, graph: dict[int, list[tuple[int, float]]], my_vertices: list[int], dist: list[float]
    ) -> list[float]:
        """Relax outgoing edges of this rank's owned vertices.

        :param graph: Adjacency list for the graph.
        :param my_vertices: Vertices owned by this rank.
        :param dist: The current globally agreed distance vector.
        :returns: A new distance vector with local relaxations applied.
        """
        local_updates = dist[:]
        for u in my_vertices:
            if dist[u] == INF:
                continue
            for v, weight in graph[u]:
                candidate = dist[u] + weight
                if candidate < local_updates[v]:
                    local_updates[v] = candidate
        return local_updates

    def _combine_distances(self, local_updates: list[float]) -> list[float]:
        """Combine every rank's local distance updates with an element-wise min.

        :param local_updates: This rank's locally relaxed distance vector.
        :returns: The globally agreed, element-wise minimum distance vector.
        """
        send_buf = np.array(local_updates, dtype="float64")
        recv_buf = np.empty_like(send_buf)
        self._comm.Allreduce(send_buf, recv_buf, op=MPI.MIN)
        result: list[float] = recv_buf.tolist()
        return result

    def _has_converged(self, new_dist: list[float], dist: list[float]) -> bool:
        """Check across all ranks whether any distance changed this round.

        :param new_dist: The distance vector produced this round.
        :param dist: The distance vector from the previous round.
        :returns: ``True`` if at least one rank observed a change.
        """
        changed_local = new_dist != dist
        has_changed: bool = self._comm.allreduce(changed_local, op=MPI.LOR)
        return has_changed


def main() -> None:
    """Entry point: relax :data:`GRAPH` from :data:`SOURCE` in parallel."""
    comm: Comm = MPI.COMM_WORLD
    solver = ParallelDijkstraSolver(comm)

    result = solver.run(GRAPH, SOURCE, NUM_VERTICES)

    if result is not None:
        print(f"Converged after {result.rounds} synchronization rounds.")
        for vertex, distance in enumerate(result.distances):
            print(f"  shortest distance to {vertex}: {distance}")


if __name__ == "__main__":
    main()
