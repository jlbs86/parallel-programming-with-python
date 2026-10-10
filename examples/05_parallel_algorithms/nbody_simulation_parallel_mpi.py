"""Parallel N-Body gravitational simulation using mpi4py (ring all-to-all).

Theory
------
In an N-body simulation, every body exerts a force on every other body, so
a naive implementation is O(n^2) per timestep. The standard distributed
parallelization is **data decomposition with a communication ring**:

1. Bodies are partitioned into contiguous blocks, one block per process.
2. In each timestep, every process computes the forces between the bodies
   *it owns* and every other body in the system. To see every other body
   exactly once without an O(P^2) all-to-all, bodies are passed around a
   **logical ring**: at each of ``P`` ring steps, every process sends its
   current working block to the next rank and receives a block from the
   previous rank (``Sendrecv``), accumulating partial forces as blocks
   pass through.
3. After each process has seen every other block exactly once, it has the
   total force on its own bodies and can integrate their velocity and
   position locally (no further communication needed for integration).

This ring communication pattern is a classic alternative to all-to-all
communication and is widely used in particle simulations, molecular
dynamics (e.g. GROMACS-style domain decomposition), and other N-body
problems where every pair must be considered.

Run with:
    mpiexec -n 4 python nbody_simulation_parallel_mpi.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mpi4py import MPI
from mpi4py.MPI import Comm

G = 6.674e-11
SOFTENING = 1e-3  # avoids force blow-up when two bodies get very close
DT = 0.01
STEPS = 5

Position = tuple[float, float]


@dataclass
class Body:
    """A point mass with position, velocity and accumulated force.

    :param mass: The mass of the body.
    :param position: The ``[x, y]`` position of the body.
    :param velocity: The ``[vx, vy]`` velocity of the body.
    :param force: The ``[fx, fy]`` force currently accumulated on the body.
    """

    mass: float
    position: list[float]
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])
    force: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def reset_force(self) -> None:
        """Zero out the accumulated force before a new timestep."""
        self.force[0] = 0.0
        self.force[1] = 0.0

    def accumulate_force_from(self, other: Body) -> None:
        """Add the gravitational force exerted by ``other`` onto this body.

        :param other: The body exerting a gravitational pull on this one.
        """
        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]
        dist_sq = dx * dx + dy * dy + SOFTENING
        dist = dist_sq**0.5
        magnitude = G * self.mass * other.mass / dist_sq
        self.force[0] += magnitude * dx / dist
        self.force[1] += magnitude * dy / dist

    def integrate(self, dt: float) -> None:
        """Advance velocity and position by one timestep using the current force.

        :param dt: The timestep size.
        """
        ax = self.force[0] / self.mass
        ay = self.force[1] / self.mass
        self.velocity[0] += ax * dt
        self.velocity[1] += ay * dt
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt


class SequentialBodyFactory:
    """Builds a synthetic set of bodies for the N-body demo.

    :param count: The number of bodies to generate.
    :param seed: The seed for the random generator.
    """

    def __init__(self, count: int, seed: int) -> None:
        self._count = count
        self._rng = random.Random(seed)

    def build(self) -> list[Body]:
        """Generate :attr:`_count` bodies with random mass and position.

        :returns: A list of newly constructed :class:`Body` instances.
        """
        return [
            Body(
                mass=self._rng.uniform(1e10, 1e12),
                position=[self._rng.uniform(-100, 100), self._rng.uniform(-100, 100)],
            )
            for _ in range(self._count)
        ]


@dataclass(frozen=True)
class NBodySimulationResult:
    """Final result of a distributed N-body simulation, available on rank 0.

    :param steps: The number of timesteps simulated.
    :param positions: The final ``(x, y)`` position of every body, rounded
        for deterministic display.
    """

    steps: int
    positions: list[Position]


class ParallelNBodySimulation:
    """Ring-based distributed N-body simulation coordinator over MPI.

    :param comm: The MPI communicator whose ranks each own a contiguous
        block of bodies and exchange them around a logical ring each
        timestep to compute all pairwise forces.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, bodies: list[Body] | None, steps: int, dt: float) -> NBodySimulationResult | None:
        """Simulate ``steps`` timesteps of gravitational interaction.

        :param bodies: The full set of bodies. Only meaningful on rank 0;
            every other rank should pass ``None``.
        :param steps: The number of timesteps to simulate.
        :param dt: The timestep size used for integration.
        :returns: An :class:`NBodySimulationResult` on rank 0, ``None``
            elsewhere.
        """
        if self._rank == 0:
            assert bodies is not None  # caller contract: only rank 0 passes real data
            chunks = self._split_into_chunks(bodies)
        else:
            chunks = None
        my_bodies = self._comm.scatter(chunks, root=0)

        for _ in range(steps):
            self._ring_step_forces(my_bodies)
            for body in my_bodies:
                body.integrate(dt)

        final_positions = self._comm.gather(
            [(round(b.position[0], 3), round(b.position[1], 3)) for b in my_bodies], root=0
        )

        if self._rank != 0:
            return None

        assert final_positions is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        return self._combine_results(steps, final_positions)

    def _split_into_chunks(self, bodies: list[Body]) -> list[list[Body]]:
        """Split ``bodies`` into :attr:`_size` interleaved chunks for scattering.

        :param bodies: The full set of bodies.
        :returns: One chunk of bodies per rank.
        """
        return [bodies[i :: self._size] for i in range(self._size)]

    def _ring_step_forces(self, my_bodies: list[Body]) -> None:
        """Accumulate one timestep of pairwise forces via ring-passed blocks.

        :param my_bodies: The bodies owned by this rank; forces are reset
            and accumulated in place.
        """
        next_rank = (self._rank + 1) % self._size
        prev_rank = (self._rank - 1) % self._size

        for body in my_bodies:
            body.reset_force()

        incoming_block = my_bodies
        for _ in range(self._size):
            # Every body in `incoming_block` currently visiting this rank
            # contributes to the force on every locally-owned body.
            for local_body in my_bodies:
                for visiting_body in incoming_block:
                    if visiting_body is not local_body:
                        local_body.accumulate_force_from(visiting_body)

            # Pass the block one step around the ring for the next rotation.
            incoming_block = self._comm.sendrecv(incoming_block, dest=next_rank, source=prev_rank)

    def _combine_results(
        self, steps: int, final_positions: list[list[Position]]
    ) -> NBodySimulationResult:
        """Flatten per-rank final positions into the final result.

        :param steps: The number of timesteps simulated.
        :param final_positions: Per-rank final body positions, in rank order.
        :returns: The combined :class:`NBodySimulationResult`.
        """
        flattened = [position for group in final_positions for position in group]
        return NBodySimulationResult(steps=steps, positions=flattened)


def main() -> None:
    """Entry point: build random bodies on rank 0 and simulate them in parallel."""
    comm: Comm = MPI.COMM_WORLD
    simulation = ParallelNBodySimulation(comm)

    bodies: list[Body] | None = None
    if simulation.rank == 0:
        bodies = SequentialBodyFactory(count=16, seed=11).build()

    result = simulation.run(bodies, STEPS, DT)

    if result is not None:
        print(f"Positions after {result.steps} steps across {comm.Get_size()} ranks:")
        for i, position in enumerate(result.positions):
            print(f"  body {i}: {position}")


if __name__ == "__main__":
    main()
