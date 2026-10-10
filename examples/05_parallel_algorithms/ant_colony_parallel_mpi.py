"""Parallel Ant Colony Optimization (ACO) for TSP using mpi4py.

Theory
------
Ant Colony Optimization is a metaheuristic where a population of "ants"
builds candidate solutions probabilistically, guided by a shared pheromone
matrix that is reinforced based on solution quality. The ant construction
phase is **embarrassingly parallel** (each ant's tour is independent given
the current pheromone matrix); only the pheromone update is a
synchronization point. This is the standard **parallel ACO** scheme:

1. The pheromone matrix is broadcast to every process at the start of each
   iteration (``bcast``).
2. Each process constructs tours for its share of the ant colony fully in
   parallel (no communication during construction).
3. Local best tours and local pheromone deltas are combined with
   ``gather``/``reduce`` so rank 0 can apply the global pheromone update
   (evaporation + reinforcement) before broadcasting the updated matrix for
   the next iteration.

This mirrors how swarm-intelligence and genetic algorithms are generally
parallelized: parallel "exploration", synchronized "exploitation" update.

Run with:
    mpiexec -n 4 python ant_colony_parallel_mpi.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

# Symmetric distance matrix for a small TSP instance (5 cities).
DISTANCES: list[list[float]] = [
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0],
]
NUM_CITIES = len(DISTANCES)
ANTS_PER_RANK = 8
ITERATIONS = 20
EVAPORATION = 0.5
ALPHA, BETA = 1.0, 3.0  # pheromone vs. heuristic (1/distance) influence

Tour = list[int]
TourWithLength = tuple[Tour, float]
PheromoneMatrix = list[list[float]]


class SequentialAntTourBuilder:
    """Builds probabilistic tours for a share of the ant colony on one rank.

    :param distances: The symmetric distance matrix between cities.
    :param rng: A per-rank random generator for independent tour construction.
    """

    def __init__(self, distances: list[list[float]], rng: random.Random) -> None:
        self._distances = distances
        self._rng = rng

    def build_tour(self, pheromone: PheromoneMatrix) -> Tour:
        """Probabilistically construct a single tour guided by ``pheromone``.

        :param pheromone: The current pheromone matrix.
        :returns: A full tour visiting every city exactly once.
        """
        num_cities = len(self._distances)
        unvisited = list(range(num_cities))
        current = self._rng.choice(unvisited)
        tour = [current]
        unvisited.remove(current)

        while unvisited:
            weights = []
            for city in unvisited:
                tau = pheromone[current][city] ** ALPHA
                distance = self._distances[current][city]
                eta = (1.0 / distance) ** BETA if distance else 1e9
                weights.append(tau * eta)
            total = sum(weights)
            probs = (
                [w / total for w in weights] if total > 0 else [1 / len(unvisited)] * len(unvisited)
            )
            next_city = self._rng.choices(unvisited, weights=probs, k=1)[0]
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        return tour

    def tour_length(self, tour: Tour) -> float:
        """Compute the total length of a closed ``tour``.

        :param tour: A full tour visiting every city exactly once.
        :returns: The sum of edge distances, including the return edge.
        """
        n = len(tour)
        return sum(self._distances[tour[i]][tour[(i + 1) % n]] for i in range(n))

    def build_colony(self, pheromone: PheromoneMatrix, ants: int) -> list[TourWithLength]:
        """Build ``ants`` tours and pair each with its length.

        :param pheromone: The current pheromone matrix.
        :param ants: The number of ants (tours) to construct.
        :returns: A list of ``(tour, length)`` pairs.
        """
        tours = [self.build_tour(pheromone) for _ in range(ants)]
        return [(tour, self.tour_length(tour)) for tour in tours]


@dataclass(frozen=True)
class AntColonyResult:
    """Final result of a distributed ACO run, available on rank 0.

    :param best_tour: The best tour found across all iterations and ranks.
    :param best_length: The length of :attr:`best_tour`.
    """

    best_tour: Tour | None
    best_length: float


class ParallelAntColonyCoordinator:
    """Distributed ACO coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to build tours
        and periodically synchronize the shared pheromone matrix.
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
        distances: list[list[float]],
        iterations: int,
        ants_per_rank: int,
    ) -> AntColonyResult | None:
        """Run ACO for ``iterations`` rounds across the communicator.

        :param distances: The symmetric distance matrix between cities.
        :param iterations: The number of ACO iterations to run.
        :param ants_per_rank: The number of ants constructed per rank per
            iteration.
        :returns: An :class:`AntColonyResult` on rank 0, ``None`` elsewhere.
        """
        num_cities = len(distances)
        pheromone: PheromoneMatrix = [[1.0] * num_cities for _ in range(num_cities)]
        rng = random.Random(1000 + self._rank)
        builder = SequentialAntTourBuilder(distances, rng)

        best_tour: Tour | None = None
        best_length = float("inf")

        for _ in range(iterations):
            pheromone = self._comm.bcast(pheromone, root=0)

            colony = builder.build_colony(pheromone, ants_per_rank)
            local_tours = [tour for tour, _ in colony]
            local_best = min(colony, key=lambda tl: tl[1])

            all_bests = self._comm.gather(local_best, root=0)
            all_tours = self._comm.gather(local_tours, root=0)

            if self._rank == 0:
                assert all_tours is not None  # guaranteed on rank 0 after `gather(..., root=0)`
                assert all_bests is not None  # guaranteed on rank 0 after `gather(..., root=0)`
                flattened_tours = [t for group in all_tours for t in group]
                pheromone = self._update_pheromone(pheromone, flattened_tours, builder)

                iteration_best_tour, iteration_best_length = min(all_bests, key=lambda tb: tb[1])
                if iteration_best_length < best_length:
                    best_length = iteration_best_length
                    best_tour = iteration_best_tour

        if self._rank != 0:
            return None

        return AntColonyResult(best_tour=best_tour, best_length=best_length)

    def _update_pheromone(
        self,
        pheromone: PheromoneMatrix,
        tours: list[Tour],
        builder: SequentialAntTourBuilder,
    ) -> PheromoneMatrix:
        """Evaporate and reinforce the pheromone matrix using every tour.

        :param pheromone: The pheromone matrix before this iteration's update.
        :param tours: Every tour constructed this iteration, across all ranks.
        :param builder: Used to compute each tour's length for reinforcement.
        :returns: The updated pheromone matrix.
        """
        updated = [[(1 - EVAPORATION) * p for p in row] for row in pheromone]
        for tour in tours:
            deposit = 1.0 / builder.tour_length(tour)
            for i in range(len(tour)):
                a, b = tour[i], tour[(i + 1) % len(tour)]
                updated[a][b] += deposit
                updated[b][a] += deposit
        return updated


def main() -> None:
    """Entry point: run ACO on the toy TSP instance in parallel."""
    comm: Comm = MPI.COMM_WORLD
    coordinator = ParallelAntColonyCoordinator(comm)

    result = coordinator.run(DISTANCES, ITERATIONS, ANTS_PER_RANK)

    if result is not None:
        print(f"Best tour found: {result.best_tour} with length {result.best_length}")


if __name__ == "__main__":
    main()
