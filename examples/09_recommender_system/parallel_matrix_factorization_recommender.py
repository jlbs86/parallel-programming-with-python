"""A real, reusable parallel Recommender System via matrix factorization.

Theory
------
Collaborative-filtering recommender systems (as popularized by the Netflix
Prize) model a sparse ``user x item`` ratings matrix ``R`` as the product
of two dense low-rank matrices::

    R ~= U @ V^T

where ``U`` is ``(num_users, k)`` (user latent factors) and ``V`` is
``(num_items, k)`` (item latent factors). Training alternates between:

1. Fixing ``V`` and solving for ``U`` (ridge-regularized least squares,
   one independent row per user).
2. Fixing ``U`` and solving for ``V`` (same, one independent row per
   item).

This is **Alternating Least Squares (ALS)**, and both steps are
embarrassingly parallel in the same way as
``matrix_multiplication_parallel_mpi.py``: every row of ``U`` (or ``V``)
is solved completely independently of every other row, given the other
factor matrix held fixed. That means the heavy linear algebra --
computing ``V^T @ V`` once per half-iteration and then one small linear
solve per user/item row -- can be split across worker processes with
zero inter-worker communication during the row-solving phase, exactly
the "broadcast the shared data, scatter the independent rows, gather the
results" pattern used throughout this repository.

Design goals of this module
----------------------------
- **Portable parallelism**: uses :mod:`multiprocessing.Pool`, part of the
  Python standard library, so this runs on *any* machine (no MPI runtime
  or GPU required) while still giving real multi-core speedup -- the
  shared-memory analogue of the MPI examples elsewhere in this repo.
- **Real parallel matrix multiplication**: the prediction step
  (``U @ V^T``) and the per-iteration ``V^T @ V`` / ``U^T @ U`` Gram
  matrix computations are the actual bottleneck of ALS at scale, and are
  parallelized by row-block decomposition, mirroring
  ``matrix_multiplication_parallel_mpi.py``.
- **OOP, typed, documented**: a small number of focused classes with
  type hints and Sphinx-style docstrings, consistent with every other
  example in this repository.

Run with::

    python parallel_matrix_factorization_recommender.py

.. note::
   On macOS with Apple's Accelerate BLAS backend (the default for many
   NumPy installs on Apple Silicon), you may see spurious
   ``RuntimeWarning: overflow/invalid value encountered in matmul``
   warnings during matrix multiplication even though the computed values
   are correct (no actual ``NaN``/``Inf`` is produced -- verified via
   ``np.isnan``/``np.isinf``). This is a known Accelerate quirk and does
   not occur with OpenBLAS (the default on most Linux distributions).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class RecommenderConfig:
    """Hyperparameters for Alternating Least Squares (ALS) training.

    :param num_factors: The rank ``k`` of the latent factor matrices.
    :param num_iterations: The number of alternating U/V update rounds.
    :param regularization: The ridge regression regularization strength.
    :param num_workers: The number of worker processes to use for row
        solves. Defaults to the number of logical CPU cores.
    :param random_seed: Seed for reproducible factor initialization.
    """

    num_factors: int = 10
    num_iterations: int = 15
    regularization: float = 0.1
    num_workers: int = os.cpu_count() or 1
    random_seed: int = 42


@dataclass(frozen=True)
class RecommendationResult:
    """Final trained recommender state and evaluation metrics.

    :param user_factors: The trained ``(num_users, k)`` user factor matrix.
    :param item_factors: The trained ``(num_items, k)`` item factor matrix.
    :param train_rmse_per_iteration: Training RMSE recorded after each ALS iteration.
    """

    user_factors: FloatMatrix
    item_factors: FloatMatrix
    train_rmse_per_iteration: list[float]

    def predict_all(self) -> FloatMatrix:
        """Reconstruct the full dense ratings prediction matrix.

        :returns: The ``(num_users, num_items)`` matrix ``U @ V^T``.
        """
        return self.user_factors @ self.item_factors.T

    def recommend_for_user(
        self, user_index: int, already_rated: set[int], top_n: int = 5
    ) -> list[tuple[int, float]]:
        """Recommend the top-N highest-predicted, not-yet-rated items for a user.

        :param user_index: The row index of the target user.
        :param already_rated: Item indices to exclude (already rated/seen).
        :param top_n: How many recommendations to return.
        :returns: ``(item_index, predicted_rating)`` pairs, best first.
        """
        predicted = self.item_factors @ self.user_factors[user_index]
        ranked_items = sorted(
            (i for i in range(len(predicted)) if i not in already_rated),
            key=lambda i: predicted[i],
            reverse=True,
        )
        return [(i, float(predicted[i])) for i in ranked_items[:top_n]]


def _solve_single_row(
    row_index: int,
    ratings_row_or_col: FloatMatrix,
    observed_mask: NDArray[np.bool_],
    fixed_factors: FloatMatrix,
    regularization: float,
    num_factors: int,
) -> FloatMatrix:
    """Solve the regularized least-squares problem for one user or item row.

    This is the per-row unit of work distributed across worker processes
    by :class:`ParallelAlsRecommender`. It is a module-level function
    (not a method) because :class:`multiprocessing.Pool` must be able to
    pickle the callable sent to each worker.

    :param row_index: Index of the user (or item) row being solved. Not
        used in the computation itself, but kept for traceability/logging
        and to keep the worker's call signature self-describing.
    :param ratings_row_or_col: The observed ratings for this row (already
        sliced to just this user's ratings across items, or this item's
        ratings across users).
    :param observed_mask: Boolean mask marking which entries of
        ``ratings_row_or_col`` are observed (nonzero) ratings.
    :param fixed_factors: The *other* factor matrix, held fixed this half-iteration.
    :param regularization: Ridge regression regularization strength.
    :param num_factors: The latent factor dimensionality ``k``.
    :returns: The updated latent factor vector for this row, shape ``(k,)``.
    """
    del row_index  # unused: present only to keep each task self-describing
    observed_factors = fixed_factors[observed_mask]
    observed_ratings = ratings_row_or_col[observed_mask]

    if observed_factors.shape[0] == 0:
        return np.zeros(num_factors)

    gram = observed_factors.T @ observed_factors
    gram += regularization * np.eye(num_factors)
    rhs = observed_factors.T @ observed_ratings
    return np.linalg.solve(gram, rhs).astype(np.float64)


class ParallelRowSolver:
    """Distributes independent per-row least-squares solves across worker processes.

    :param pool: An active :class:`multiprocessing.pool.Pool` to submit work to.
    """

    def __init__(self, pool: mp.pool.Pool) -> None:
        self._pool = pool

    def solve_all_rows(
        self,
        ratings: FloatMatrix,
        observed_mask: NDArray[np.bool_],
        fixed_factors: FloatMatrix,
        regularization: float,
        num_factors: int,
    ) -> FloatMatrix:
        """Solve every row's least-squares problem in parallel.

        :param ratings: The ratings matrix, one row per entity being solved for.
        :param observed_mask: Boolean mask of observed entries, same shape as ``ratings``.
        :param fixed_factors: The other factor matrix, held fixed this call.
        :param regularization: Ridge regression regularization strength.
        :param num_factors: The latent factor dimensionality ``k``.
        :returns: The updated factor matrix, shape ``(ratings.shape[0], num_factors)``.
        """
        worker = partial(
            _solve_single_row,
            fixed_factors=fixed_factors,
            regularization=regularization,
            num_factors=num_factors,
        )
        # Each row is fully independent given `fixed_factors`, so rows are
        # scattered across the worker pool with zero inter-worker
        # communication -- the shared-memory analogue of `comm.scatter`.
        rows = self._pool.starmap(
            worker,
            ((i, ratings[i], observed_mask[i]) for i in range(ratings.shape[0])),
        )
        return np.vstack(rows).astype(np.float64)


class ParallelMatrixMultiplier:
    """Row-block-parallel dense matrix multiplication via a worker pool.

    Mirrors the row-decomposition strategy of
    ``matrix_multiplication_parallel_mpi.py``, but using
    :class:`multiprocessing.Pool` workers instead of MPI ranks, so the
    same algorithmic idea runs without requiring an MPI installation.

    :param pool: An active :class:`multiprocessing.pool.Pool` to submit work to.
    """

    def __init__(self, pool: mp.pool.Pool) -> None:
        self._pool = pool

    def multiply(self, a: FloatMatrix, b: FloatMatrix) -> FloatMatrix:
        """Compute ``a @ b`` by splitting ``a``'s rows across worker processes.

        :param a: The left-hand matrix, shape ``(n, k)``.
        :param b: The right-hand matrix, shape ``(k, m)``.
        :returns: The product, shape ``(n, m)``.
        """
        num_workers = self._pool._processes  # type: ignore[attr-defined]
        row_chunks = np.array_split(a, min(num_workers, a.shape[0]) or 1, axis=0)
        results = self._pool.starmap(_multiply_chunk, ((chunk, b) for chunk in row_chunks))
        return np.vstack(results)


def _multiply_chunk(a_chunk: FloatMatrix, b: FloatMatrix) -> FloatMatrix:
    """Multiply one row-block of ``a`` by the full matrix ``b`` (one worker's share)."""
    return a_chunk @ b


class ParallelAlsRecommender:
    """Trains a matrix-factorization recommender with parallel ALS.

    :param config: Training hyperparameters.
    """

    def __init__(self, config: RecommenderConfig | None = None) -> None:
        self._config = config or RecommenderConfig()

    def fit(self, ratings: FloatMatrix) -> RecommendationResult:
        """Train user/item latent factors on an observed ratings matrix.

        :param ratings: A ``(num_users, num_items)`` matrix where unobserved
            entries are ``0.0`` and observed ratings are positive.
        :returns: The trained :class:`RecommendationResult`.
        """
        config = self._config
        rng = np.random.default_rng(config.random_seed)
        num_users, num_items = ratings.shape
        observed_mask = ratings != 0.0

        user_factors = rng.normal(scale=0.1, size=(num_users, config.num_factors))
        item_factors = rng.normal(scale=0.1, size=(num_items, config.num_factors))

        rmse_history: list[float] = []
        with mp.Pool(processes=config.num_workers) as pool:
            solver = ParallelRowSolver(pool)
            multiplier = ParallelMatrixMultiplier(pool)

            for _ in range(config.num_iterations):
                user_factors = solver.solve_all_rows(
                    ratings, observed_mask, item_factors, config.regularization, config.num_factors
                )
                item_factors = solver.solve_all_rows(
                    ratings.T,
                    observed_mask.T,
                    user_factors,
                    config.regularization,
                    config.num_factors,
                )

                predictions = multiplier.multiply(user_factors, item_factors.T)
                rmse_history.append(self._rmse(ratings, predictions, observed_mask))

        return RecommendationResult(
            user_factors=user_factors,
            item_factors=item_factors,
            train_rmse_per_iteration=rmse_history,
        )

    @staticmethod
    def _rmse(ratings: FloatMatrix, predictions: FloatMatrix, mask: NDArray[np.bool_]) -> float:
        """Root-mean-squared error over observed entries only.

        :param ratings: The ground-truth ratings matrix.
        :param predictions: The reconstructed ``U @ V^T`` prediction matrix.
        :param mask: Boolean mask of observed entries.
        :returns: The RMSE restricted to observed entries.
        """
        errors = (ratings[mask] - predictions[mask]) ** 2
        return float(np.sqrt(errors.mean())) if errors.size else 0.0


class SyntheticRatingsGenerator:
    """Generates a synthetic sparse ratings matrix for demonstration purposes.

    :param num_users: Number of synthetic users.
    :param num_items: Number of synthetic items.
    :param density: Fraction of entries that are observed ratings.
    :param seed: Random seed for reproducibility.
    """

    def __init__(self, num_users: int, num_items: int, density: float, seed: int) -> None:
        self._num_users = num_users
        self._num_items = num_items
        self._density = density
        self._rng = np.random.default_rng(seed)

    def generate(self) -> FloatMatrix:
        """Build a synthetic ratings matrix from a true low-rank structure plus noise.

        :returns: A ``(num_users, num_items)`` sparse ratings matrix with
            values in ``{0, 1, 2, 3, 4, 5}`` (``0`` meaning unobserved).
        """
        true_rank = 4
        true_users = self._rng.normal(size=(self._num_users, true_rank))
        true_items = self._rng.normal(size=(self._num_items, true_rank))
        dense_scores = true_users @ true_items.T

        # Rescale to a 1-5 star range and clip.
        normalized = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        star_ratings = np.clip(np.round(normalized * 4) + 1, 1, 5)

        observed = self._rng.random(size=star_ratings.shape) < self._density
        return np.where(observed, star_ratings, 0.0)


def main() -> None:
    """Entry point: train the recommender on synthetic data and print recommendations."""
    num_users, num_items = 400, 300
    ratings = SyntheticRatingsGenerator(num_users, num_items, density=0.1, seed=7).generate()

    config = RecommenderConfig(num_factors=8, num_iterations=12, regularization=0.1)
    recommender = ParallelAlsRecommender(config)

    print(
        f"Training on {num_users}x{num_items} matrix using "
        f"{config.num_workers} worker processes..."
    )
    result = recommender.fit(ratings)

    print("Training RMSE per iteration:")
    for i, rmse in enumerate(result.train_rmse_per_iteration, start=1):
        print(f"  iteration {i:2d}: RMSE = {rmse:.4f}")

    target_user = 0
    already_rated = set(np.nonzero(ratings[target_user])[0].tolist())
    recommendations = result.recommend_for_user(target_user, already_rated, top_n=5)

    print(f"\nTop 5 recommendations for user {target_user}:")
    for item_index, predicted_rating in recommendations:
        print(f"  item {item_index}: predicted rating {predicted_rating:.2f}")


if __name__ == "__main__":
    main()
