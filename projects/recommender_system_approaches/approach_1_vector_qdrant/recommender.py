"""Approach 1: Vector-based movie recommender using Qdrant (embeddings + ANN search).

Source
------
Based on "How to Create a Vector-Based Movie Recommendation System Using
Qdrant DB" (Akriti Upadhyay, Medium):
https://medium.com/@akriti.upadhyay/how-to-create-a-vector-based-movie-recommendation-system-using-qdrant-db-e026554f49ec

Pipeline
--------
This is the most modern of the three approaches: instead of building a
dense ``n x n`` similarity matrix in memory (what Approaches 2 and 3 do),
it embeds every movie into a high-dimensional **semantic vector space**
using a transformer sentence-embedding model, stores those vectors in a
**vector database** (Qdrant), and answers "find movies similar to X" by
running an **approximate nearest neighbor (ANN) search** -- via Qdrant's
HNSW (Hierarchical Navigable Small World) index -- instead of comparing
against every other movie exhaustively.

1. **Combine content fields** (plot, genres, director, actors) into one
   text description per movie, same idea as Approach 3's "soup".
2. **Embed with a sentence-transformer**: encode each movie's text into
   a dense vector using ``sentence-transformers/all-MiniLM-L6-v2``, a
   384-dimensional transformer embedding model (small, fast, and a very
   common default choice for this kind of tutorial -- the real article
   uses the same class of embedding model from the ``sentence-transformers``
   library).
3. **Create a Qdrant collection** configured for cosine-distance vectors
   of the embedding model's dimensionality, then **upsert** every
   movie's vector together with its metadata (title, genres, etc.) as a
   "payload".
4. **Query**: to recommend movies similar to a title, fetch that movie's
   stored vector and run Qdrant's ``query_points`` (nearest-neighbor
   search) against the collection, excluding the query point itself.

Why this is different from Approaches 2 and 3
-------------------------------------------------
- Approaches 2/3 use sparse, lexical vectors (word counts / TF-IDF): two
  movies only look similar if they share literal words.
- Approach 1 uses dense *semantic* embeddings: a transformer model can
  recognize that "an astronaut stranded in space" and "a crew lost in a
  wormhole" are thematically related even without shared vocabulary.
- Approaches 2/3 compute a full similarity matrix upfront (O(n^2) space);
  a real vector database like Qdrant scales to millions of items because
  HNSW search is sub-linear in the number of indexed points.

Run with::

    python recommender.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from movies import MOVIES, Movie  # noqa: E402

COLLECTION_NAME = "movies"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


class MovieDescriptionBuilder:
    """Builds the natural-language description embedded for each movie."""

    def build(self, movie: Movie) -> str:
        """Combine plot, genres, director, and actors into one descriptive text.

        :param movie: The movie to build a description for.
        :returns: A natural-language description suitable for sentence embedding.
        """
        genre_text = ", ".join(movie.genres)
        actor_text = ", ".join(movie.actors[:3])
        return (
            f"{movie.title}. {movie.plot} "
            f"Genres: {genre_text}. Directed by {movie.director}. "
            f"Starring {actor_text}."
        )


@dataclass(frozen=True)
class Recommendation:
    """A single recommended movie with its vector similarity score.

    :param title: The recommended movie's title.
    :param similarity_score: Cosine similarity between the query and
        result embeddings, in ``[0, 1]``.
    """

    title: str
    similarity_score: float


@dataclass(frozen=True)
class VectorIndexStats:
    """Build-time statistics for the Qdrant vector index.

    :param num_movies: Number of movies embedded and indexed.
    :param embedding_dimensions: Dimensionality of each stored vector.
    :param embed_seconds: Time spent computing embeddings.
    :param index_seconds: Time spent upserting vectors into Qdrant.
    """

    num_movies: int
    embedding_dimensions: int
    embed_seconds: float
    index_seconds: float


class VectorMovieRecommender:
    """End-to-end vector-based movie recommender backed by Qdrant.

    :param movies: The catalog of movies to index and recommend from.
    :param client: An optional pre-configured :class:`QdrantClient`. If
        omitted, an in-memory Qdrant instance is used (no server required).
    """

    def __init__(self, movies: list[Movie], client: QdrantClient | None = None) -> None:
        self._movies = movies
        self._client = client or QdrantClient(location=":memory:")
        self._description_builder = MovieDescriptionBuilder()
        self._encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._title_to_id = {movie.title: i for i, movie in enumerate(movies)}

    def fit(self) -> VectorIndexStats:
        """Embed every movie and index the vectors in a Qdrant collection.

        :returns: Build-time statistics as :class:`VectorIndexStats`.
        """
        descriptions = [self._description_builder.build(movie) for movie in self._movies]

        embed_start = time.perf_counter()
        embeddings = self._encoder.encode(descriptions, show_progress_bar=False)
        embed_seconds = time.perf_counter() - embed_start

        index_start = time.perf_counter()
        if self._client.collection_exists(COLLECTION_NAME):
            self._client.delete_collection(COLLECTION_NAME)
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"title": movie.title, "genres": movie.genres},
            )
            for i, movie in enumerate(self._movies)
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)
        index_seconds = time.perf_counter() - index_start

        return VectorIndexStats(
            num_movies=len(self._movies),
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            embed_seconds=embed_seconds,
            index_seconds=index_seconds,
        )

    def recommend(self, title: str, top_n: int = 5) -> list[Recommendation]:
        """Recommend the top-N nearest-neighbor movies to ``title`` by embedding similarity.

        :param title: The query movie's exact title.
        :param top_n: How many recommendations to return.
        :returns: Up to ``top_n`` :class:`Recommendation` objects, best match first.
        :raises ValueError: If ``title`` is not found in the catalog.
        """
        if title not in self._title_to_id:
            raise ValueError(f"Movie {title!r} not found in catalog.")

        query_id = self._title_to_id[title]
        retrieved = self._client.retrieve(
            collection_name=COLLECTION_NAME, ids=[query_id], with_vectors=True
        )[0]
        query_vector = retrieved.vector
        if not isinstance(query_vector, list):
            raise ValueError(f"Expected a plain vector for {title!r}, got {type(query_vector)}.")

        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_n + 1,  # +1 because the query movie itself will match with score 1.0
        ).points

        recommendations = []
        for point in results:
            if point.id == query_id:
                continue
            payload = point.payload or {}
            recommendations.append(Recommendation(payload["title"], point.score))
            if len(recommendations) == top_n:
                break
        return recommendations


def main() -> None:
    """Entry point: build the vector index and print sample recommendations."""
    recommender = VectorMovieRecommender(MOVIES)
    stats = recommender.fit()
    print(
        f"Embedded {stats.num_movies} movies into {stats.embedding_dimensions}-dim vectors "
        f"in {stats.embed_seconds:.4f}s, indexed in Qdrant in {stats.index_seconds:.4f}s\n"
    )

    for query in ["The Avengers", "Interstellar", "The Godfather"]:
        print(f"Recommendations for {query!r}:")
        for rec in recommender.recommend(query, top_n=5):
            print(f"  {rec.title} (similarity={rec.similarity_score:.3f})")
        print()


if __name__ == "__main__":
    main()
