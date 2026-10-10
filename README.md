# Parallel Programming with Python

A comprehensive, hands-on course repository for learning **parallel and
distributed programming in Python**, covering distributed-memory message
passing (`mpi4py`), shared-memory parallelism (`multiprocessing` and
Numba's OpenMP-equivalent `prange`), GPU parallelism (CUDA via
`numba.cuda`), and a complete, real-world parallel recommender system.

Every example is implemented with modern, idiomatic Python: object-oriented
design, full type hints, [Sphinx](https://www.sphinx-doc.org/)-style
docstrings, and clean linting (`ruff` + `mypy`). Every example has been
executed and verified, not just written.

## Why this repository exists

Most introductions to parallel programming either stay purely theoretical
or hand you toy snippets that don't resemble production code. This
repository instead pairs **the theory** behind each parallel pattern
(data decomposition, task decomposition, embarrassingly parallel
workloads, reductions, collective communication, sorting networks,
recursive communicator splitting, GPU thread hierarchies, shared-memory
threading) with **a fully working, typed, tested implementation** of that
pattern applied to a well-known algorithm.

## Repository structure

```text
examples/
├── 05_parallel_algorithms/   # mpi4py: 16 classic algorithms, parallelized
├── 06_cuda/                  # GPU parallelism via numba.cuda
├── 07_openmp_equivalent/     # Shared-memory CPU parallelism via Numba prange
├── 08_gpu_setup_linux/       # NVIDIA driver + CUDA Toolkit setup guide (Linux)
└── 09_recommender_system/    # Real parallel recommender system (ALS + matmul)
tests/
└── parallel_testing/         # pytest-xdist: running the test suite across CPU cores
docs/
└── source/                   # Sphinx documentation (autodoc from docstrings)
```

## Parallel algorithms covered (`examples/05_parallel_algorithms/`)

Every module below is a standalone, runnable MPI program built around an
OOP coordinator class (`Parallel<X>Coordinator`/`Solver`/`Classifier`)
that wraps `mpi4py.MPI.Comm`, plus a `Sequential<X>` class for the local
per-rank computation.

| Algorithm | File | Parallel strategy |
|---|---|---|
| Merge Sort | `merge_sort_parallel_mpi.py` | Divide-and-conquer: scatter chunks, sort locally, k-way merge |
| Quicksort (Hyperquicksort) | `quick_sort_parallel_mpi.py` | Recursive communicator splitting (`Comm.Split`) |
| Bubble Sort (Odd-Even Transposition) | `bubble_sort_parallel_mpi.py` | Sorting network over point-to-point `Sendrecv` |
| Breadth-First Search (BFS) | `bfs_parallel_mpi.py` | Frontier decomposition + `Allgather` |
| Depth-First Search (DFS) | `dfs_parallel_mpi.py` | Task-parallel independent-subtree exploration |
| Dijkstra's Shortest Path | `dijkstra_parallel_mpi.py` | Vertex partitioning + relax-until-convergence `Allreduce` |
| Parallel Search (linear & binary) | `parallel_search_mpi.py` | Data decomposition + logical-OR `Allreduce` |
| Matrix Multiplication | `matrix_multiplication_parallel_mpi.py` | Row-block decomposition |
| Fast Fourier Transform (FFT) | `fft_parallel_mpi.py` | Recursive Cooley-Tukey mapped onto communicator splitting |
| N-Body Simulation | `nbody_simulation_parallel_mpi.py` | Ring communication pattern |
| Monte Carlo Pi Estimation | `monte_carlo_pi_parallel_mpi.py` | Embarrassingly parallel + single `reduce` |
| Prime Sieve (segmented) | `prime_sieve_parallel_mpi.py` | Segmented Sieve of Eratosthenes over disjoint ranges |
| K-Nearest Neighbors (KNN) | `knn_parallel_mpi.py` | Data-parallel distance computation + bounded top-k reduction |
| Ant Colony Optimization (TSP) | `ant_colony_parallel_mpi.py` | Parallel exploration + synchronized pheromone update |
| MapReduce (word count) | `mapreduce_parallel_mpi.py` | Partition / Map / Shuffle (`alltoall`) / Reduce |
| Parallel Reduce & Scan | `parallel_reduce_scan_mpi.py` | `MPI_Reduce`/`Scan`/`Exscan` plus a manual recursive-doubling scan |

Run any example with:

```bash
mpiexec -n 4 python examples/05_parallel_algorithms/<file>.py
```

## Shared-memory parallelism: the OpenMP equivalent (`examples/07_openmp_equivalent/`)

`prange_shared_memory_numba.py` demonstrates Python's answer to OpenMP's
`#pragma omp parallel for`: Numba's `@njit(parallel=True)` plus
`numba.prange`, which JIT-compiles Python to native code and distributes
loop iterations across real OS threads with the GIL released. Includes a
benchmark comparing sequential vs. parallel elementwise transforms,
reductions, and matrix multiplication.

```bash
python examples/07_openmp_equivalent/prange_shared_memory_numba.py
```

## GPU parallelism with CUDA (`examples/06_cuda/`)

`cuda_vector_and_matrix_kernels.py` implements vector addition and dense
matrix multiplication as real CUDA kernels using `numba.cuda`, following
the grid/block/thread hierarchy that underlies all GPU programming. The
script detects whether a CUDA-capable GPU is present and explains how to
get one if not.

See **`examples/08_gpu_setup_linux/README.md`** for a complete guide to
setting up an NVIDIA GPU for CUDA development on Linux: driver
installation, CUDA Toolkit setup, Python environment configuration, and
a troubleshooting table.

## Parallel Recommender System (`examples/09_recommender_system/`)

`parallel_matrix_factorization_recommender.py` is a complete,
**production-shaped** collaborative-filtering recommender system trained
with Alternating Least Squares (ALS) matrix factorization. It uses real
parallelism throughout:

- Per-user and per-item least-squares row solves are distributed across
  a `multiprocessing.Pool`, since each row is solved completely
  independently of every other row.
- The prediction step (`U @ V^T`) is computed with a from-scratch
  **parallel matrix multiplication** (row-block decomposition across
  worker processes), the same algorithmic pattern as
  `matrix_multiplication_parallel_mpi.py`, but implemented with the
  standard library's `multiprocessing` instead of MPI.

This design requires **no MPI installation and no GPU** -- it runs on any
machine with Python and multiple CPU cores, making it directly reusable
in other projects.

```bash
python examples/09_recommender_system/parallel_matrix_factorization_recommender.py
```

## Testing in parallel with `pytest-xdist` (`tests/parallel_testing/`)

The test suite validates the sequential building blocks behind every MPI
example (no MPI runtime required to run the tests themselves) and is
designed to scale across CPU cores:

```bash
pip install pytest pytest-xdist
pytest -n auto tests/parallel_testing/
```

See `tests/parallel_testing/README.md` for the theory behind test-suite
parallelization and the available distribution strategies.

## Documentation

Full API documentation is generated with Sphinx directly from this
repository's docstrings:

```bash
pip install -r docs/requirements.txt
python -m sphinx -b html docs/source docs/build
```

Open `docs/build/index.html` to browse the generated reference, including
the conceptual overview in `docs/source/theory.rst`.

## Setup

### Core dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
```

### MPI examples

Requires an MPI implementation (e.g. [Open MPI](https://www.open-mpi.org/)
or [MPICH](https://www.mpich.org/)) and `mpi4py`:

```bash
# macOS
brew install open-mpi
# Ubuntu/Debian
sudo apt install -y libopenmpi-dev openmpi-bin

pip install mpi4py
```

### Shared-memory / OpenMP-equivalent and CUDA examples

```bash
pip install numba
```

CUDA examples additionally require an NVIDIA GPU -- see
`examples/08_gpu_setup_linux/README.md`.

### Development tools

```bash
pip install -e ".[dev,docs]"
```

This installs `pytest`, `pytest-xdist`, `ruff`, and `mypy`, plus Sphinx
for documentation. Lint and type-check with:

```bash
ruff check examples/ tests/
mypy examples/
```

## License

GNU General Public License v3.0 -- see `LICENSE`.
