"""Parallel K-Nearest Neighbors (KNN) classification using mpi4py.

Theory
------
Classifying a query point with KNN requires computing its distance to
*every* point in the training set, then taking a vote among the ``k``
closest. Distance computation is independent per training point, making
this a textbook **data-parallel reduction**:

1. Rank 0 scatters the training set in contiguous chunks (``scatter``).
2. Every rank computes distances from the (broadcast) query point to only
   its local chunk -- pure data parallelism, no communication needed here.
3. Each rank keeps only its local top-``k`` candidates (a partial
   reduction done locally to minimize communication volume).
4. Local top-k lists are gathered at rank 0 and merged into the global
   top-k, which decides the predicted label by majority vote.

This "map locally, keep a bounded partial result, reduce centrally"
pattern is the same one used by distributed similarity search and
large-scale recommender systems.

Run with:
    mpiexec -n 4 python knn_parallel_mpi.py
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

K = 5

Point = tuple[float, float]
LabeledPoint = tuple[Point, str]


class SequentialKNearestNeighborsFinder:
    """Finds the local top-k nearest labeled points to a query point.

    :param dataset: The labeled points this rank is responsible for.
    :param k: The number of nearest neighbors to keep.
    """

    def __init__(self, dataset: list[LabeledPoint], k: int) -> None:
        self._dataset = dataset
        self._k = k

    def nearest_to(self, query: Point) -> list[LabeledPoint]:
        """Return the ``k`` points in :attr:`_dataset` closest to ``query``.

        :param query: The point to measure distances from.
        :returns: Up to ``k`` labeled points, sorted by increasing distance.
        """
        return heapq.nsmallest(
            self._k, self._dataset, key=lambda item: self._euclidean(item[0], query)
        )

    @staticmethod
    def _euclidean(a: Point, b: Point) -> float:
        """Euclidean distance between two 2-D points.

        :param a: The first point.
        :param b: The second point.
        :returns: The Euclidean distance between ``a`` and ``b``.
        """
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class SyntheticDatasetBuilder:
    """Builds a synthetic two-cluster labeled dataset for the KNN demo.

    :param size: The number of points to generate.
    :param seed: The seed for the random generator.
    """

    def __init__(self, size: int, seed: int) -> None:
        self._size = size
        self._rng = random.Random(seed)

    def build(self) -> list[LabeledPoint]:
        """Generate the synthetic dataset.

        :returns: A list of ``(point, label)`` pairs drawn from two
            Gaussian clusters labeled ``"A"`` and ``"B"``.
        """
        dataset: list[LabeledPoint] = []
        for _ in range(self._size):
            if self._rng.random() < 0.5:
                point = (self._rng.gauss(0, 1), self._rng.gauss(0, 1))
                label = "A"
            else:
                point = (self._rng.gauss(5, 1), self._rng.gauss(5, 1))
                label = "B"
            dataset.append((point, label))
        return dataset


@dataclass(frozen=True)
class KNearestNeighborsResult:
    """Final result of a distributed KNN classification, available on rank 0.

    :param query: The query point that was classified.
    :param neighbors: The global top-k nearest labeled points.
    :param prediction: The majority-vote predicted label.
    """

    query: Point
    neighbors: list[LabeledPoint]
    prediction: str


class ParallelKNearestNeighborsClassifier:
    """Distributed KNN classifier coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate to classify a
        query point against a distributed training set.
    """

    def __init__(self, comm: Comm, k: int = K) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self._k = k

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(
        self, dataset: list[LabeledPoint] | None, query: Point | None
    ) -> KNearestNeighborsResult | None:
        """Classify ``query`` against ``dataset`` distributed across ranks.

        :param dataset: The full training set. Only meaningful on rank 0;
            every other rank should pass ``None``.
        :param query: The query point. Only meaningful on rank 0; every
            other rank should pass ``None``.
        :returns: A :class:`KNearestNeighborsResult` on rank 0, ``None``
            elsewhere.
        """
        query = self._comm.bcast(query, root=0)
        if self._rank == 0:
            assert dataset is not None  # caller contract: only rank 0 passes real data
            chunks = self._split_into_chunks(dataset)
        else:
            chunks = None
        my_chunk = self._comm.scatter(chunks, root=0)

        local_top_k = SequentialKNearestNeighborsFinder(my_chunk, self._k).nearest_to(query)
        gathered = self._comm.gather(local_top_k, root=0)

        if self._rank != 0:
            return None

        assert gathered is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        return self._combine_results(query, gathered)

    def _split_into_chunks(self, dataset: list[LabeledPoint]) -> list[list[LabeledPoint]]:
        """Split ``dataset`` into :attr:`_size` interleaved chunks for scattering.

        :param dataset: The full training set.
        :returns: One chunk of labeled points per rank.
        """
        return [dataset[i :: self._size] for i in range(self._size)]

    def _combine_results(
        self, query: Point, gathered: list[list[LabeledPoint]]
    ) -> KNearestNeighborsResult:
        """Merge per-rank top-k lists into the global top-k and vote on a label.

        :param query: The query point that was classified.
        :param gathered: Per-rank local top-k lists.
        :returns: The combined :class:`KNearestNeighborsResult`.
        """
        candidates = [item for group in gathered for item in group]
        global_top_k = SequentialKNearestNeighborsFinder(candidates, self._k).nearest_to(query)

        votes: dict[str, int] = {}
        for _, label in global_top_k:
            votes[label] = votes.get(label, 0) + 1
        prediction = max(votes, key=lambda label: votes[label])

        return KNearestNeighborsResult(query=query, neighbors=global_top_k, prediction=prediction)


def main() -> None:
    """Entry point: build a synthetic dataset on rank 0 and classify a query point."""
    comm: Comm = MPI.COMM_WORLD
    classifier = ParallelKNearestNeighborsClassifier(comm)

    dataset: list[LabeledPoint] | None = None
    query: Point | None = None
    if classifier.rank == 0:
        dataset = SyntheticDatasetBuilder(200, seed=99).build()
        query = (2.5, 2.5)

    result = classifier.run(dataset, query)

    if result is not None:
        print(f"Query point: {result.query}")
        print(f"{K}-nearest neighbors: {result.neighbors}")
        print(f"Predicted label: {result.prediction}")


if __name__ == "__main__":
    main()
