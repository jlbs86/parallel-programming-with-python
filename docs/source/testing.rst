Testing in Parallel with ``pytest-xdist``
============================================

See ``tests/parallel_testing/README.md`` for the full guide on running
this repository's test suite across multiple CPU cores with
``pytest-xdist`` to reduce test execution time.

Quick reference::

    pip install pytest pytest-xdist
    pytest -n auto tests/parallel_testing/
