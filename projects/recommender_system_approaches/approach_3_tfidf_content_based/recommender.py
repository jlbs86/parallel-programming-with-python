"""Approach 3: TF-IDF content-based movie recommender (web-mining-is688 style).

Source
------
Based on the "web-mining-is688-spring-2021" Medium series on content-based
film recommendation, e.g. "Film Recommendation System Using Content-Based
Filtering":
https://medium.com/web-mining-is688-spring-2021/film-recommendation-system-using-content-based-filtering-eb856c7359f3

and the related overview article "Movie Recommendation System":
https://medium.com/web-mining-is688-spring-2021/movie-recommendation-system-bb46ba0f6f86

Pipeline
--------
Content-based filtering recommends items similar to ones a user already
liked, based purely on item *attributes* -- no other users' behavior is
needed (unlike collaborative filtering). The standard content-based
pipeline for text attributes is:

1. **Combine content fields** (here: plot, genres, director, actors)
   into a single "soup" of text per movie.
2. **TF-IDF vectorization**: unlike the raw Bag-of-Words count used in
   Approach 2, Term Frequency-Inverse Document Frequency down-weights
   words that appear in *many* movies (common, less distinctive words)
   and up-weights words that are frequent in one movie but rare across
   the whole catalog -- making the resulting vectors more discriminative
   for similarity search.
3. **Cosine similarity** between every pair of TF-IDF vectors, exactly as
   in Approach 2, but computed on TF-IDF-weighted vectors instead of raw
   counts.
4. **Recommendation**: look up the query movie's row, sort by
   descending similarity, return the top-N (excluding itself).

Where this differs from Approach 2 (NLP + Bag-of-Words)
----------------------------------------------------------
- Approach 2 uses RAKE for plot keyword extraction and raw word *counts*
  (:class:`~sklearn.feature_extraction.text.CountVectorizer`).
- Approach 3 uses the full plot text (with standard English stopword
  removal) and TF-IDF *weights*
  (:class:`~sklearn.feature_extraction.text.TfidfVectorizer`), which is
  the technique emphasized across the web-mining-is688 series and is the
  most common choice in production content-based systems (also used by
  the classic "TMDB 5000 movies" Kaggle tutorials this series is based on).

Run with::

    python recommender.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from movies import MOVIES, Movie  # noqa: E402


class ContentSoupBuilder:
    """Builds the per-movie 'content soup' string used for TF-IDF vectorization."""

    def build(self, movie: Movie) -> str:
        """Combine plot, genres, director, and actors into one text blob.

        Unlike Approach 2's token-merging strategy, this approach keeps
        the plot as natural-language text (TF-IDF's stopword removal and
        inverse-document-frequency weighting make RAKE-style keyword
        pre-extraction unnecessary) and repeats structured fields with
        light normalization to keep them matchable as tokens.

        :param movie: The movie to build a content representation for.
        :returns: A single text blob combining all content fields.
        """
        genre_text = " ".join(movie.genres)
        actor_text = " ".join(a.replace(" ", "") for a in movie.actors[:3])
        director_text = movie.director.replace(" ", "")
        return f"{movie.plot} {genre_text} {director_text} {actor_text}"


@dataclass(frozen=True)
class Recommendation:
    """A single recommended movie with its similarity score.

    :param title: The recommended movie's title.
    :param similarity_score: Cosine similarity to the query movie, in ``[0, 1]``.
    """

    title: str
    similarity_score: float


@dataclass(frozen=True)
class TfidfIndex:
    """The fitted TF-IDF similarity matrix and title index for recommendations.

    :param titles: Movie titles in the same row order as ``similarity_matrix``.
    :param similarity_matrix: The ``n x n`` cosine similarity matrix.
    :param vocabulary_size: Number of distinct terms in the fitted TF-IDF vocabulary.
    :param build_seconds: Wall-clock time taken to build this index.
    """

    titles: list[str]
    similarity_matrix: list[list[float]]
    vocabulary_size: int
    build_seconds: float


class TfidfContentBasedRecommender:
    """End-to-end TF-IDF content-based movie recommender.

    :param movies: The catalog of movies to index and recommend from.
    """

    def __init__(self, movies: list[Movie]) -> None:
        self._movies = movies
        self._soup_builder = ContentSoupBuilder()
        self._index: TfidfIndex | None = None

    def fit(self) -> TfidfIndex:
        """Build the TF-IDF vectors and the cosine similarity matrix.

        :returns: The fitted :class:`TfidfIndex`.
        """
        start = time.perf_counter()

        content_soups = [self._soup_builder.build(movie) for movie in self._movies]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(content_soups)

        # linear_kernel on L2-normalized TF-IDF vectors is mathematically
        # equivalent to cosine similarity, but noticeably faster since it
        # skips the redundant norm computation cosine_similarity repeats.
        similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

        self._index = TfidfIndex(
            titles=[movie.title for movie in self._movies],
            similarity_matrix=similarity_matrix.tolist(),
            vocabulary_size=len(vectorizer.vocabulary_),
            build_seconds=time.perf_counter() - start,
        )
        return self._index

    def recommend(self, title: str, top_n: int = 5) -> list[Recommendation]:
        """Recommend the top-N most similar movies to ``title``.

        :param title: The query movie's exact title.
        :param top_n: How many recommendations to return.
        :returns: Up to ``top_n`` :class:`Recommendation` objects, best match first.
        :raises ValueError: If the index has not been built yet, or
            ``title`` is not found in the catalog.
        """
        if self._index is None:
            raise ValueError("Call fit() before recommend().")
        if title not in self._index.titles:
            raise ValueError(f"Movie {title!r} not found in catalog.")

        idx = self._index.titles.index(title)
        scores = list(enumerate(self._index.similarity_matrix[idx]))
        scores.sort(key=lambda pair: pair[1], reverse=True)

        recommendations = []
        for other_idx, score in scores:
            if other_idx == idx:
                continue
            recommendations.append(Recommendation(self._index.titles[other_idx], score))
            if len(recommendations) == top_n:
                break
        return recommendations


def main() -> None:
    """Entry point: build the index and print sample recommendations."""
    recommender = TfidfContentBasedRecommender(MOVIES)
    index = recommender.fit()
    print(
        f"Built {len(index.titles)}x{len(index.titles)} TF-IDF similarity matrix "
        f"({index.vocabulary_size} vocabulary terms) in {index.build_seconds:.4f}s\n"
    )

    for query in ["The Avengers", "Interstellar", "The Godfather"]:
        print(f"Recommendations for {query!r}:")
        for rec in recommender.recommend(query, top_n=5):
            print(f"  {rec.title} (similarity={rec.similarity_score:.3f})")
        print()


if __name__ == "__main__":
    main()
