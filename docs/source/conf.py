"""Sphinx configuration for the Parallel Programming with Python documentation.

See https://www.sphinx-doc.org/en/master/usage/configuration.html for the
full list of configuration options.
"""

from __future__ import annotations

import os
import sys

# Make every example package importable so autodoc can introspect classes
# and generate API reference pages directly from their docstrings.
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../examples/05_parallel_algorithms"))
sys.path.insert(0, os.path.abspath("../../examples/06_cuda"))
sys.path.insert(0, os.path.abspath("../../examples/07_openmp_equivalent"))
sys.path.insert(0, os.path.abspath("../../examples/09_recommender_system"))

project = "Parallel Programming with Python"
copyright = "2024-2026, jlbs86 and course contributors"
author = "jlbs86"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

# This project writes Sphinx-style (reStructuredText field list) docstrings
# directly, e.g. `:param x:` / `:returns:`, which autodoc understands
# natively. napoleon is enabled in case any Google/NumPy-style docstrings
# are added later (e.g. in third-party code referenced via intersphinx).
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
