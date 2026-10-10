Parallel Algorithms (``mpi4py``)
===================================

Every module below lives in ``examples/05_parallel_algorithms/`` and is
runnable with ``mpiexec -n N python <module>.py``. Each module's own
docstring (reproduced here via autodoc) explains the specific parallel
decomposition strategy it uses.

Sorting
--------

.. automodule:: merge_sort_parallel_mpi
   :members:

.. automodule:: quick_sort_parallel_mpi
   :members:

.. automodule:: bubble_sort_parallel_mpi
   :members:

Graph algorithms
------------------

.. automodule:: bfs_parallel_mpi
   :members:

.. automodule:: dfs_parallel_mpi
   :members:

.. automodule:: dijkstra_parallel_mpi
   :members:

Search
-------

.. automodule:: parallel_search_mpi
   :members:

Numerical and scientific computing
-------------------------------------

.. automodule:: matrix_multiplication_parallel_mpi
   :members:

.. automodule:: fft_parallel_mpi
   :members:

.. automodule:: nbody_simulation_parallel_mpi
   :members:

.. automodule:: monte_carlo_pi_parallel_mpi
   :members:

.. automodule:: prime_sieve_parallel_mpi
   :members:

Machine learning and optimization
-------------------------------------

.. automodule:: knn_parallel_mpi
   :members:

.. automodule:: ant_colony_parallel_mpi
   :members:

Data processing primitives
-----------------------------

.. automodule:: mapreduce_parallel_mpi
   :members:

.. automodule:: parallel_reduce_scan_mpi
   :members:
