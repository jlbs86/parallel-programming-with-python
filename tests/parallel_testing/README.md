# Running the Test Suite in Parallel with `pytest-xdist`

## Theory

Test suites are an ideal workload for data parallelism: individual test
functions are (by design) independent of one another, so running them
concurrently across CPU cores produces the same results as running them
sequentially, just faster. `pytest-xdist` parallelizes test execution using
a **master/worker (controller/subprocess) model**, conceptually similar to
the MPI "scatter work, gather results" examples elsewhere in this
repository, but implemented with OS processes and `execnet` instead of
message passing:

1. The controller process collects the full list of test items.
2. Work is distributed to `N` worker subprocesses, one per requested core,
   using one of several distribution strategies (see below).
3. Each worker imports the test modules and executes its assigned tests in
   an isolated interpreter process (true parallelism, not limited by the
   GIL, the same reason `multiprocessing` is used for CPU-bound shared
   memory parallelism in `examples/04_shared_memory_multiprocessing/`).
4. Results stream back to the controller, which merges them into a single
   report.

## Distribution strategies

- `-n auto`: spawn as many workers as there are logical CPU cores.
- `-n 4`: spawn exactly 4 workers.
- `--dist=load` (default): tests are handed out dynamically as workers
  become free -- best when test durations are uneven.
- `--dist=loadscope`: all tests in the same module/class run on the same
  worker -- required when tests share expensive fixture state that is not
  safe to duplicate (e.g. spinning up one `MPI.COMM_WORLD`-like resource
  per module).
- `--dist=each`: every worker runs the *entire* suite -- useful for testing
  across multiple Python versions/environments simultaneously, not for
  speed.

## Setup

```bash
pip install pytest pytest-xdist
```

## Usage

Run the whole suite using all available cores:

```bash
pytest -n auto
```

Run with a fixed worker count (e.g. on a shared CI runner with 4 cores):

```bash
pytest -n 4
```

Keep tests that share module-level state together on one worker:

```bash
pytest -n auto --dist=loadscope
```

## Notes on testing MPI-based code

The MPI examples in `examples/01`-`examples/05_parallel_algorithms/` spawn
their *own* parallelism via `mpiexec -n N`. Running `pytest-xdist` with
multiple workers does **not** parallelize what happens *inside* a single
`mpiexec` invocation; instead, `test_algorithms_parallel.py` in this
directory tests the *sequential* building-block classes (e.g.
`SequentialMergeSorter`, `SequentialDepthFirstSearcher`) directly, in
isolation from MPI. This is intentional: unit tests should validate
algorithmic correctness without depending on an MPI runtime being
installed, while `mpiexec` is reserved for actually exercising the
distributed coordination logic end-to-end.
