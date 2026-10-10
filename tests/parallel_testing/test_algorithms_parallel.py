"""Unit tests for the sequential building blocks behind the parallel examples.

These tests are written to run efficiently under ``pytest-xdist`` (see
``README.md`` in this directory): every test is fully independent, has no
shared mutable state, and does not require an MPI runtime, so they can be
freely distributed across worker processes with::

    pytest -n auto tests/parallel_testing/

Each test targets the local ``Sequential<X>`` class extracted from the
corresponding example in ``examples/05_parallel_algorithms/``, so the
algorithmic correctness of the building block used inside each MPI
coordinator is verified without needing ``mpiexec``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "05_parallel_algorithms"
sys.path.insert(0, str(EXAMPLES_DIR))

from bubble_sort_parallel_mpi import SequentialOddEvenSorter  # noqa: E402
from dijkstra_parallel_mpi import GRAPH as DIJKSTRA_GRAPH  # noqa: E402
from knn_parallel_mpi import (  # noqa: E402
    SequentialKNearestNeighborsFinder,
    SyntheticDatasetBuilder,
)
from merge_sort_parallel_mpi import SequentialMergeSorter  # noqa: E402
from parallel_search_mpi import (  # noqa: E402
    SequentialBinarySearcher,
    SequentialLinearSearcher,
)
from prime_sieve_parallel_mpi import SequentialPrimeSieve  # noqa: E402
from quick_sort_parallel_mpi import SequentialQuickSorter  # noqa: E402


class TestSequentialMergeSorter:
    """Tests for the local merge-sort building block."""

    @pytest.mark.parametrize(
        "values",
        [
            [],
            [1],
            [3, 1, 2],
            [5, 4, 3, 2, 1],
            [-10, 10, 0, -5, 5],
        ],
    )
    def test_sort_matches_builtin(self, values: list[int]) -> None:
        sorter = SequentialMergeSorter(values)
        assert sorter.sort() == sorted(values)


class TestSequentialOddEvenSorter:
    """Tests for the odd-even transposition sort building block."""

    def test_compare_and_keep_min(self) -> None:
        sorter = SequentialOddEvenSorter(7)
        assert sorter.compare_and_keep_min(3) == 3
        assert sorter.value == 3

    def test_compare_and_keep_max(self) -> None:
        sorter = SequentialOddEvenSorter(2)
        assert sorter.compare_and_keep_max(9) == 9
        assert sorter.value == 9

    def test_value_unchanged_when_already_extreme(self) -> None:
        min_sorter = SequentialOddEvenSorter(1)
        assert min_sorter.compare_and_keep_min(5) == 1

        max_sorter = SequentialOddEvenSorter(9)
        assert max_sorter.compare_and_keep_max(2) == 9


class TestSequentialQuickSorter:
    """Tests for the local quicksort building block."""

    @pytest.mark.parametrize(
        "values",
        [[], [1], [4, 2, 7, 1, 9, 3], [-1, -1, -1], list(range(20, 0, -1))],
    )
    def test_sort_matches_builtin(self, values: list[int]) -> None:
        sorter = SequentialQuickSorter(values)
        assert sorter.sort() == sorted(values)


class TestSequentialPrimeSieve:
    """Tests for the base-range and segmented prime sieve building blocks."""

    def test_base_primes(self) -> None:
        sieve = SequentialPrimeSieve(limit=30)
        assert sieve.sieve_base_primes() == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_segment_matches_full_sieve(self) -> None:
        base_primes = SequentialPrimeSieve(limit=10).sieve_base_primes()
        assert base_primes == [2, 3, 5, 7]

        full = SequentialPrimeSieve(limit=30)
        segment_primes = full.sieve_segment(11, 30, base_primes)
        expected_segment = [p for p in full.sieve_base_primes() if p > 10]
        assert segment_primes == expected_segment == [11, 13, 17, 19, 23, 29]


class TestSequentialSearchers:
    """Tests for linear and binary search building blocks."""

    def test_linear_search_finds_all_matches(self) -> None:
        data = [1, 5, 3, 5, 9, 5]
        searcher = SequentialLinearSearcher(data)
        assert searcher.find_local_matches(5, offset=100) == [101, 103, 105]

    def test_linear_search_no_match(self) -> None:
        searcher = SequentialLinearSearcher([1, 2, 3])
        assert searcher.find_local_matches(99, offset=0) == []

    def test_binary_search_hit_and_miss(self) -> None:
        searcher = SequentialBinarySearcher([10, 20, 30, 40, 50])
        assert searcher.find(30) == 2
        assert searcher.find(31) is None

    def test_binary_search_bounds(self) -> None:
        assert SequentialBinarySearcher([10, 20, 30]).bounds == (10, 30)
        assert SequentialBinarySearcher([]).bounds is None


class TestKNearestNeighbors:
    """Tests for the KNN local finder and synthetic dataset builder."""

    def test_nearest_to_returns_k_closest(self) -> None:
        dataset = [((0.0, 0.0), "A"), ((10.0, 10.0), "B"), ((0.1, 0.1), "A")]
        finder = SequentialKNearestNeighborsFinder(dataset, k=2)
        nearest = finder.nearest_to((0.0, 0.0))
        assert len(nearest) == 2
        assert nearest[0][1] == "A"

    def test_synthetic_dataset_builder_is_deterministic(self) -> None:
        first = SyntheticDatasetBuilder(50, seed=1).build()
        second = SyntheticDatasetBuilder(50, seed=1).build()
        assert first == second
        assert len(first) == 50


class TestDijkstraGraphFixture:
    """Sanity checks on the fixed graph used by the parallel Dijkstra example."""

    def test_graph_has_no_negative_weights(self) -> None:
        for edges in DIJKSTRA_GRAPH.values():
            for _, weight in edges:
                assert weight >= 0

    def test_source_vertex_exists(self) -> None:
        assert 0 in DIJKSTRA_GRAPH
