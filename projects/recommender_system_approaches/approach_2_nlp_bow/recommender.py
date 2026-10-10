"""Approach 2: NLP + Bag-of-Words movie recommender (Analytics Vidhya style).

Source
------
Based on "Movie Recommendation Engine with NLP" (Analytics Vidhya, 2022):
https://www.analyticsvidhya.com/blog/2022/01/movie-recommendation-engine-with-nlp/

Pipeline
--------
1. **Keyword extraction with RAKE** (Rapid Automatic Keyword Extraction):
   extract the most relevant keywords from each movie's plot synopsis,
   discarding stopwords and ranking by co-occurrence degree.
2. **Feature engineering**: merge each actor's/director's first and last
   name into a single token (e.g. ``"christophernolan"``) so the model
   never confuses two different people who happen to share a first name,
   and lowercase everything for consistent matching.
3. **Bag of Words**: concatenate genre tags, director, lead actors, and
   extracted plot keywords into one "bag of words" string per movie.
4. **Vectorization**: convert every movie's bag of words into a count
   vector with :class:`~sklearn.feature_extraction.text.CountVectorizer`.
5. **Similarity**: compute pairwise cosine similarity between every pair
   of movies' count vectors, producing an ``n x n`` similarity matrix.
6. **Recommendation**: for a query title, look up its row in the
   similarity matrix, sort scores in descending order, and return the
   top-N most similar movies (excluding the query itself).

This mirrors the original tutorial's five sections: "Importing the
Dependencies and Loading the Data", "Text Preprocessing with NLP",
"Generating Word Representations using Bag Of Words", "Vectorizing Words
and Creating the Similarity Matrix", and "Training and Testing Our
Recommendation Engine" -- translated into reusable, typed OOP classes.

Run with::

    python recommender.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from movies import MOVIES, Movie  # noqa: E402


def _merge_name(name: str) -> str:
    """Collapse a person's name into one lowercase token, e.g. ``"Tom Hanks"`` -> ``"tomhanks"``.

    This prevents the model from treating two different people who share
    a first name (e.g. two different "Chris"es) as the same entity.

    :param name: A person's full name.
    :returns: The name as a single lowercase token with no whitespace.
    """
    return name.lower().replace(" ", "")


class KeywordExtractor:
    """Extracts ranked keywords from plot text using RAKE.

    :param rake: A configured :class:`rake_nltk.Rake` instance. A new one
        is created if not provided.
    """

    def __init__(self, rake: Rake | None = None) -> None:
        self._rake = rake or Rake()

    def extract(self, plot: str) -> list[str]:
        """Extract keywords from a plot synopsis, ranked by RAKE's degree score.

        :param plot: The plot synopsis text.
        :returns: Keywords extracted from ``plot``, stopwords removed.
        """
        self._rake.extract_keywords_from_text(plot)
        scores = self._rake.get_word_degrees()
        return list(scores.keys())


class BagOfWordsBuilder:
    """Builds the per-movie 'bag of words' string used for vectorization.

    :param keyword_extractor: The :class:`KeywordExtractor` used to pull
        keywords out of each movie's plot.
    """

    def __init__(self, keyword_extractor: KeywordExtractor | None = None) -> None:
        self._keyword_extractor = keyword_extractor or KeywordExtractor()

    def build(self, movie: Movie) -> str:
        """Combine genre, director, actors, and plot keywords into one string.

        :param movie: The movie to build a bag-of-words representation for.
        :returns: A single whitespace-separated string of tokens.
        """
        genre_tokens = [_merge_name(g) for g in movie.genres]
        director_tokens = [_merge_name(movie.director)]
        actor_tokens = [_merge_name(a) for a in movie.actors[:3]]
        keyword_tokens = self._keyword_extractor.extract(movie.plot)

        all_tokens = genre_tokens + director_tokens + actor_tokens + keyword_tokens
        return " ".join(all_tokens)


@dataclass(frozen=True)
class Recommendation:
    """A single recommended movie with its similarity score.

    :param title: The recommended movie's title.
    :param similarity_score: Cosine similarity to the query movie, in ``[0, 1]``.
    """

    title: str
    similarity_score: float


@dataclass(frozen=True)
class BagOfWordsIndex:
    """The fitted vectorizer, similarity matrix, and title index for recommendations.

    :param titles: Movie titles in the same row order as ``similarity_matrix``.
    :param similarity_matrix: The ``n x n`` cosine similarity matrix.
    :param build_seconds: Wall-clock time taken to build this index.
    """

    titles: list[str]
    similarity_matrix: list[list[float]]
    build_seconds: float


class NlpBagOfWordsRecommender:
    """End-to-end NLP + Bag-of-Words content-based movie recommender.

    :param movies: The catalog of movies to index and recommend from.
    """

    def __init__(self, movies: list[Movie]) -> None:
        self._movies = movies
        self._bow_builder = BagOfWordsBuilder()
        self._index: BagOfWordsIndex | None = None

    def fit(self) -> BagOfWordsIndex:
        """Build the bag-of-words vectors and the cosine similarity matrix.

        :returns: The fitted :class:`BagOfWordsIndex`.
        """
        start = time.perf_counter()

        bags_of_words = [self._bow_builder.build(movie) for movie in self._movies]
        count_vectorizer = CountVectorizer()
        count_matrix = count_vectorizer.fit_transform(bags_of_words)
        similarity_matrix = cosine_similarity(count_matrix, count_matrix)

        self._index = BagOfWordsIndex(
            titles=[movie.title for movie in self._movies],
            similarity_matrix=similarity_matrix.tolist(),
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
    recommender = NlpBagOfWordsRecommender(MOVIES)
    index = recommender.fit()
    print(
        f"Built {len(index.titles)}x{len(index.titles)} similarity matrix "
        f"in {index.build_seconds:.4f}s\n"
    )

    for query in ["The Avengers", "Interstellar", "The Godfather"]:
        print(f"Recommendations for {query!r}:")
        for rec in recommender.recommend(query, top_n=5):
            print(f"  {rec.title} (similarity={rec.similarity_score:.3f})")
        print()


if __name__ == "__main__":
    main()
