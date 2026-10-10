Theory of Parallel Programming
================================

This repository teaches parallel programming through four complementary
models, each with its own trade-offs:

Distributed-memory parallelism (MPI / ``mpi4py``)
---------------------------------------------------

Multiple **processes**, each with its own private memory, communicate
explicitly by sending messages (point-to-point: ``send``/``recv``, or
collective: ``broadcast``/``scatter``/``gather``/``reduce``/``scan``).
This model scales from a single multi-core laptop to thousands of nodes
in a supercomputing cluster, because it makes no assumption about shared
memory. See ``examples/01_point_to_point/`` through
``examples/05_parallel_algorithms/``.

Shared-memory parallelism (threads / processes / Numba)
-----------------------------------------------------------

Multiple **threads or processes on one machine** share (or can share)
the same memory space. Python's Global Interpreter Lock (GIL) prevents
plain threads from running CPU-bound bytecode concurrently, so this
repository uses two complementary approaches:

- :mod:`multiprocessing` for true multi-core parallelism of ordinary
  Python code (used by the recommender system in
  ``examples/09_recommender_system/``).
- Numba's ``@njit(parallel=True)`` + :func:`numba.prange`, which compiles
  Python functions to native code and parallelizes loops across threads
  with the GIL released -- the direct Python analogue of OpenMP's
  ``#pragma omp parallel for``. See ``examples/07_openmp_equivalent/``.

GPU parallelism (CUDA via ``numba.cuda``)
--------------------------------------------

Thousands of lightweight GPU threads, organized into blocks and grids,
execute the same kernel code over massive, simple, data-parallel
workloads. See ``examples/06_cuda/`` and the Linux GPU setup guide in
``examples/08_gpu_setup_linux/``.

Choosing a model
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Model
     - Scales to
     - Best for
   * - MPI (``mpi4py``)
     - Many nodes/cluster
     - Large distributed workloads
   * - ``multiprocessing``
     - One machine's CPU cores
     - Arbitrary Python logic
   * - Numba ``prange`` (OpenMP)
     - One machine's CPU cores
     - Numeric kernels, tight loops
   * - CUDA (``numba.cuda``)
     - One GPU's thousands of cores
     - Massively data-parallel, arithmetically simple kernels
