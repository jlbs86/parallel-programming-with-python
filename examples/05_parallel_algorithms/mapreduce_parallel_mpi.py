"""Parallel MapReduce (Spark-style transformations) using mpi4py.

Theory
------
MapReduce / Spark express data processing as two phases:

- **Map**: apply a pure function independently to every element of a
  distributed dataset. Embarrassingly parallel by construction.
- **Reduce** (or "reduceByKey" in Spark terms): group mapped results by
  key and combine the values for each key with an associative operator.
  This requires a **shuffle**: records with the same key must end up on
  the same process before they can be combined.

This example implements a minimal but real distributed word-count
pipeline -- the "hello world" of MapReduce -- using only mpi4py
primitives, mirroring what a cluster computing framework does internally:

1. **Partition**: the input documents are scattered across ranks
   (data decomposition).
2. **Map**: each rank independently tokenizes its local documents into
   ``(word, 1)`` pairs.
3. **Shuffle**: every rank buckets its pairs by ``hash(key) % size`` and
   the buckets are exchanged with ``alltoall``, so afterward every rank
   holds *all* records for the keys it is responsible for -- the single
   genuinely communication-heavy step in MapReduce.
4. **Reduce**: each rank sums the counts for the keys it now owns
   locally.
5. **Collect**: results are gathered back at rank 0.

Run with:
    mpiexec -n 4 python mapreduce_parallel_mpi.py
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from mpi4py import MPI
from mpi4py.MPI import Comm

DOCUMENTS: list[str] = [
    "parallel programming with python and mpi4py",
    "shared memory parallelism with threads and processes",
    "message passing parallelism with mpi4py across many nodes",
    "python makes parallel and distributed programming approachable",
]

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")

WordCountPair = tuple[str, int]


class SequentialWordCountMapper:
    """Pure map step: tokenizes documents into ``(word, 1)`` pairs.

    :param documents: The local chunk of documents to tokenize.
    """

    def __init__(self, documents: list[str]) -> None:
        self._documents = documents

    def map(self) -> list[WordCountPair]:
        """Tokenize every document into ``(word, 1)`` pairs.

        :returns: A flat list of ``(word, 1)`` pairs.
        """
        pairs: list[WordCountPair] = []
        for doc in self._documents:
            for word in WORD_RE.findall(doc.lower()):
                pairs.append((word, 1))
        return pairs


class SequentialWordCountReducer:
    """Combines ``(key, value)`` pairs sharing a key -- the 'reduceByKey'.

    :param pairs: The ``(key, value)`` pairs owned by this rank after shuffling.
    """

    def __init__(self, pairs: list[WordCountPair]) -> None:
        self._pairs = pairs

    def reduce(self) -> Counter[str]:
        """Sum values sharing the same key.

        :returns: A :class:`collections.Counter` mapping word to count.
        """
        counts: Counter[str] = Counter()
        for key, value in self._pairs:
            counts[key] += value
        return counts


@dataclass(frozen=True)
class MapReduceResult:
    """Final result of a distributed MapReduce word count, available on rank 0.

    :param word_counts: The combined word counts across the whole corpus.
    """

    word_counts: Counter[str]


class ParallelMapReduceCoordinator:
    """Distributed MapReduce coordinator built on top of an MPI communicator.

    :param comm: The MPI communicator whose ranks cooperate through
        partition, map, shuffle, and reduce phases.
    """

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

    @property
    def rank(self) -> int:
        """The rank of the local process within :attr:`_comm`."""
        return self._rank

    def run(self, documents: list[str] | None) -> MapReduceResult | None:
        """Run the full partition/map/shuffle/reduce/collect pipeline.

        :param documents: The full input corpus. Only meaningful on rank 0;
            every other rank should pass ``None``.
        :returns: A :class:`MapReduceResult` on rank 0, ``None`` elsewhere.
        """
        if self._rank == 0:
            assert documents is not None  # caller contract: only rank 0 passes real data
            chunks = self._split_into_chunks(documents)
        else:
            chunks = None
        my_documents = self._comm.scatter(chunks, root=0)

        local_pairs = SequentialWordCountMapper(my_documents).map()
        owned_pairs = self._shuffle(local_pairs)
        local_counts = SequentialWordCountReducer(owned_pairs).reduce()

        gathered = self._comm.gather(dict(local_counts), root=0)

        if self._rank != 0:
            return None

        assert gathered is not None  # guaranteed on rank 0 after `gather(..., root=0)`
        return self._combine_results(gathered)

    def _split_into_chunks(self, documents: list[str]) -> list[list[str]]:
        """Split ``documents`` into :attr:`_size` interleaved chunks.

        :param documents: The full input corpus.
        :returns: One chunk of documents per rank.
        """
        return [documents[i :: self._size] for i in range(self._size)]

    def _shuffle(self, local_pairs: list[WordCountPair]) -> list[WordCountPair]:
        """Redistribute ``(key, value)`` pairs so every key lands on one rank.

        :param local_pairs: This rank's locally mapped ``(key, value)`` pairs.
        :returns: The pairs this rank is now responsible for reducing.
        """
        buckets: list[list[WordCountPair]] = [[] for _ in range(self._size)]
        for key, value in local_pairs:
            owner = hash(key) % self._size
            buckets[owner].append((key, value))

        # All-to-all exchange: bucket[i] sent by every rank ends up on rank i.
        received_buckets = self._comm.alltoall(buckets)
        return [pair for bucket in received_buckets for pair in bucket]

    def _combine_results(self, gathered: list[dict[str, int]]) -> MapReduceResult:
        """Combine every rank's local word counts into the final result.

        :param gathered: Per-rank local word counts, in rank order.
        :returns: The combined :class:`MapReduceResult`.
        """
        final_counts: Counter[str] = Counter()
        for partial in gathered:
            final_counts.update(partial)
        return MapReduceResult(word_counts=final_counts)


def main() -> None:
    """Entry point: run the distributed word-count pipeline over :data:`DOCUMENTS`."""
    comm: Comm = MPI.COMM_WORLD
    coordinator = ParallelMapReduceCoordinator(comm)

    result = coordinator.run(DOCUMENTS)

    if result is not None:
        print("Word counts (distributed MapReduce):")
        for word, count in result.word_counts.most_common():
            print(f"  {word}: {count}")


if __name__ == "__main__":
    main()
