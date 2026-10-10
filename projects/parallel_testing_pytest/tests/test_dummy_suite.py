"""30 dummy test cases used to demonstrate parallel test execution with pytest-xdist.

Theory
------
Test suites are an ideal workload for data parallelism: each test
function is independent, so running them concurrently across CPU cores
produces identical results to running them sequentially, just faster.
`pytest-xdist` parallelizes execution using a controller/worker-process
model: the controller collects every test item, hands work out to N
worker subprocesses (one per requested core), and merges their results.

Each test below includes a deliberate, artificial `time.sleep()` to
simulate a slow test (e.g. a test that waits on I/O, a database, or a
network call) -- this is what makes the parallel-vs-sequential timing
comparison meaningful. A real suite of 30 near-instant unit tests
wouldn't show a measurable speedup, because process-launch overhead
would dominate; these dummy tests simulate a more realistic mix of
"slow" tests, which is where parallel execution actually pays off.

Run sequentially:
    pytest tests/test_dummy_suite.py

Run in parallel across all available CPU cores:
    pytest tests/test_dummy_suite.py -n auto
"""

from __future__ import annotations

import time

import pytest

# Each dummy test sleeps for this long to simulate real work (I/O, a
# network call, a slow computation, etc.). With 30 tests at 0.3s each,
# sequential execution takes roughly 30 * 0.3s = 9s, while parallel
# execution across C cores takes roughly (30 / C) * 0.3s.
SIMULATED_WORK_SECONDS = 0.3


def _do_simulated_work(seconds: float = SIMULATED_WORK_SECONDS) -> int:
    """Simulate a slow operation (e.g. I/O-bound work) and return a dummy result.

    :param seconds: How long to sleep, simulating work duration.
    :returns: A fixed dummy result value, used only so tests have
        something concrete to assert on.
    """
    time.sleep(seconds)
    return 42


class TestDummyArithmetic:
    """Dummy tests simulating slow arithmetic/validation checks."""

    def test_case_01_addition(self) -> None:
        assert _do_simulated_work() == 42
        assert 2 + 2 == 4

    def test_case_02_subtraction(self) -> None:
        assert _do_simulated_work() == 42
        assert 10 - 3 == 7

    def test_case_03_multiplication(self) -> None:
        assert _do_simulated_work() == 42
        assert 6 * 7 == 42

    def test_case_04_division(self) -> None:
        assert _do_simulated_work() == 42
        assert 20 / 4 == 5

    def test_case_05_modulo(self) -> None:
        assert _do_simulated_work() == 42
        assert 17 % 5 == 2

    def test_case_06_power(self) -> None:
        assert _do_simulated_work() == 42
        assert 2**10 == 1024

    def test_case_07_negative_numbers(self) -> None:
        assert _do_simulated_work() == 42
        assert -5 + 5 == 0

    def test_case_08_floating_point(self) -> None:
        assert _do_simulated_work() == 42
        assert abs(0.1 + 0.2 - 0.3) < 1e-9


class TestDummyStrings:
    """Dummy tests simulating slow string-processing checks."""

    def test_case_09_concatenation(self) -> None:
        assert _do_simulated_work() == 42
        assert "foo" + "bar" == "foobar"

    def test_case_10_uppercase(self) -> None:
        assert _do_simulated_work() == 42
        assert "hello".upper() == "HELLO"

    def test_case_11_lowercase(self) -> None:
        assert _do_simulated_work() == 42
        assert "WORLD".lower() == "world"

    def test_case_12_strip(self) -> None:
        assert _do_simulated_work() == 42
        assert "  padded  ".strip() == "padded"

    def test_case_13_split(self) -> None:
        assert _do_simulated_work() == 42
        assert "a,b,c".split(",") == ["a", "b", "c"]

    def test_case_14_join(self) -> None:
        assert _do_simulated_work() == 42
        assert "-".join(["x", "y", "z"]) == "x-y-z"

    def test_case_15_replace(self) -> None:
        assert _do_simulated_work() == 42
        assert "foo bar".replace("bar", "baz") == "foo baz"

    def test_case_16_startswith(self) -> None:
        assert _do_simulated_work() == 42
        assert "parallel".startswith("par")

    def test_case_17_endswith(self) -> None:
        assert _do_simulated_work() == 42
        assert "pytest".endswith("test")


class TestDummyCollections:
    """Dummy tests simulating slow collection/data-structure checks."""

    def test_case_18_list_append(self) -> None:
        assert _do_simulated_work() == 42
        values = [1, 2, 3]
        values.append(4)
        assert values == [1, 2, 3, 4]

    def test_case_19_list_sort(self) -> None:
        assert _do_simulated_work() == 42
        assert sorted([3, 1, 2]) == [1, 2, 3]

    def test_case_20_set_union(self) -> None:
        assert _do_simulated_work() == 42
        assert {1, 2} | {2, 3} == {1, 2, 3}

    def test_case_21_set_intersection(self) -> None:
        assert _do_simulated_work() == 42
        assert {1, 2, 3} & {2, 3, 4} == {2, 3}

    def test_case_22_dict_lookup(self) -> None:
        assert _do_simulated_work() == 42
        data = {"a": 1, "b": 2}
        assert data["a"] == 1

    def test_case_23_dict_merge(self) -> None:
        assert _do_simulated_work() == 42
        assert {**{"a": 1}, **{"b": 2}} == {"a": 1, "b": 2}

    def test_case_24_tuple_unpacking(self) -> None:
        assert _do_simulated_work() == 42
        a, b, c = (1, 2, 3)
        assert (a, b, c) == (1, 2, 3)

    def test_case_25_list_comprehension(self) -> None:
        assert _do_simulated_work() == 42
        assert [x * 2 for x in range(5)] == [0, 2, 4, 6, 8]


class TestDummyParametrized:
    """A dummy parametrized test demonstrating multiple parallel-friendly test items."""

    @pytest.mark.parametrize("value,expected", [(6, 36)])
    def test_case_26_square(self, value: int, expected: int) -> None:
        assert _do_simulated_work() == 42
        assert value**2 == expected


class TestDummyExceptions:
    """Dummy tests simulating slow checks that also validate error handling."""

    def test_case_27_zero_division_raises(self) -> None:
        assert _do_simulated_work() == 42
        with pytest.raises(ZeroDivisionError):
            _ = 1 / 0

    def test_case_28_key_error_raises(self) -> None:
        assert _do_simulated_work() == 42
        empty_dict: dict[str, int] = {}
        with pytest.raises(KeyError):
            _ = empty_dict["missing"]

    def test_case_29_index_error_raises(self) -> None:
        assert _do_simulated_work() == 42
        with pytest.raises(IndexError):
            [][0]

    def test_case_30_type_error_raises(self) -> None:
        assert _do_simulated_work() == 42
        with pytest.raises(TypeError):
            _ = "a" + 1  # type: ignore[operator]
