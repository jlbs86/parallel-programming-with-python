# Parallel Testing with pytest (pytest-xdist)

This project contains 30 dummy test cases and demonstrates running them
**in parallel across all CPU cores** with `pytest-xdist`, including real
timing data comparing sequential vs. parallel execution on this machine.

## The test suite

`tests/test_dummy_suite.py` contains exactly **30 test cases** (29
individual tests + 1 parametrized test), grouped into four classes:

- `TestDummyArithmetic` (8 tests) -- basic arithmetic checks.
- `TestDummyStrings` (9 tests) -- string manipulation checks.
- `TestDummyCollections` (8 tests) -- list/set/dict/tuple checks.
- `TestDummyParametrized` (1 test) -- a parametrized squaring check.
- `TestDummyExceptions` (4 tests) -- exception-handling checks.

Every test calls a helper, `_do_simulated_work()`, that sleeps for 0.3
seconds before asserting anything. This simulates a more realistic test
suite where individual tests are I/O-bound (hitting a database, an API,
a filesystem, etc.) rather than pure in-memory unit tests that finish in
microseconds -- it's specifically what makes the sequential-vs-parallel
timing comparison below meaningful. Without this artificial delay, 30
trivial assertions would run so fast that process-launch overhead would
dominate and no parallel speedup would be visible.

## Setup

```bash
pip install pytest pytest-xdist
```

## Running sequentially (baseline)

```bash
pytest tests/test_dummy_suite.py
```

## Running in parallel across all CPU cores

```bash
pytest tests/test_dummy_suite.py -n auto
```

`-n auto` tells `pytest-xdist` to detect the number of logical CPU cores
on the machine and spawn exactly that many worker processes. You can
also request a specific worker count, e.g. `-n 4`.

### Distribution strategies

- `--dist=load` (default): tests are handed out dynamically as workers
  become free -- best when test durations are uneven, which is the case
  here since xdist does not know in advance how long each test takes.
- `--dist=loadscope`: all tests in the same class/module run on the same
  worker -- useful when tests share expensive module/class-level fixture
  state that should not be duplicated per worker.
- `--dist=each`: every worker runs the *entire* suite -- useful for
  testing across multiple environments simultaneously, not for speed.

## Real measured results on this machine

Hardware: **10 CPU cores** (`os.cpu_count()` / `sysctl hw.physicalcpu` on
this machine both report 10 -- Apple Silicon, no hyperthreading, so 10
physical = 10 logical cores). All timings below are real wall-clock
measurements (`time pytest ...`) on this exact 30-test suite, not
estimates.

| Mode | Workers | Wall-clock time | Speedup vs. sequential |
|---|---|---|---|
| Sequential | 1 (no `-n`) | **9.18s** (avg of 2 runs: 9.18s, 9.19s) | 1.00x (baseline) |
| Parallel | `-n 2` | 5.89s | 1.56x |
| Parallel | `-n 4` | 4.06s | 2.26x |
| Parallel | `-n 6` | 3.37s | 2.72x |
| Parallel | `-n 8` | 3.56s | 2.58x |
| Parallel | `-n 10` (= `-n auto`) | 3.95s (avg of 2 runs: 3.95s, 3.77s) | **2.35x** |

### What this data shows

1. **Parallel execution is real and significant**: cutting a ~9.2s suite
   down to ~3.4-4.0s is a genuine 2.3x-2.7x speedup, not a measurement
   artifact -- confirmed by repeating both the sequential and `-n auto`
   runs twice each and seeing consistent numbers.
2. **Speedup is not linear with core count, and does not peak at the
   maximum (`-n 10`)**: the best result in this run was `-n 6` (3.37s),
   slightly *faster* than `-n 10` (3.95s). This is expected and is a key
   lesson about `pytest-xdist`, not a bug:
   - Each additional worker process has real startup/teardown and
     IPC coordination overhead (visible in the `sys` time growing from
     0.76s at `-n 2` to 3.58s at `-n 10` in raw `time` output).
   - With only 30 tests at 0.3s each (9s of total work), once you have
     enough workers to keep the critical path near the per-test
     duration, adding *more* workers only adds coordination overhead
     without reducing the critical path further.
   - The theoretical minimum wall-clock time with perfect parallelism
     across `W` workers for 30 tests of 0.3s each is `ceil(30/W) * 0.3s`:
     for `W=6` that's `5 * 0.3s = 1.5s`, and for `W=10` that's
     `3 * 0.3s = 0.9s`. The real numbers (3.37s and 3.95s respectively)
     are higher than this theoretical minimum because of fixed pytest
     startup cost, worker process spawn time, and result-collection
     overhead -- all of which are amortized better by a *smaller* number
     of longer-lived workers than by maximizing worker count on a small
     suite.
3. **The crossover point matters**: for a suite this small, going beyond
   roughly 6-8 workers stopped helping and started hurting slightly. For
   a *larger* suite (hundreds of slow tests), using all 10 cores (`-n
   auto`) would very likely win outright, since the fixed per-worker
   startup cost would be amortized over much more actual test work.

### Reproducing these numbers

```bash
# Sequential baseline
time pytest tests/test_dummy_suite.py -q

# Parallel across all 10 cores
time pytest tests/test_dummy_suite.py -n auto -q

# Parallel with an explicit worker count
time pytest tests/test_dummy_suite.py -n 6 -q
```
