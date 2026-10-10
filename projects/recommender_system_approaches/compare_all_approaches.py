"""Run all three movie recommender approaches side by side on the same catalog.

This script builds each of the three recommenders analyzed from the
reference articles -- vector/Qdrant embeddings, NLP Bag-of-Words, and
TF-IDF content-based filtering -- against the exact same movie dataset,
and prints their top-5 recommendations for the same set of query movies
so the techniques can be directly compared.

Run with::

    python compare_all_approaches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "approach_1_vector_qdrant"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "approach_2_nlp_bow"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "approach_3_tfidf_content_based"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

QUERIES = ["The Avengers", "Interstellar", "The Godfather"]


def main() -> None:
    """Build all three recommenders and print a side-by-side comparison."""
    from movies import MOVIES

    print("=" * 70)
    print("Approach 1: Vector-based (Qdrant + sentence-transformers embeddings)")
    print("=" * 70)
    from approach_1_vector_qdrant.recommender import VectorMovieRecommender

    vector_recommender = VectorMovieRecommender(MOVIES)
    vector_stats = vector_recommender.fit()
    print(
        f"Embedded {vector_stats.num_movies} movies into "
        f"{vector_stats.embedding_dimensions}-dim vectors in "
        f"{vector_stats.embed_seconds:.4f}s, indexed in {vector_stats.index_seconds:.4f}s\n"
    )
    for query in QUERIES:
        print(f"  {query}:")
        for rec in vector_recommender.recommend(query, top_n=5):
            print(f"    - {rec.title} ({rec.similarity_score:.3f})")
    print()

    print("=" * 70)
    print("Approach 2: NLP + Bag-of-Words (RAKE + CountVectorizer)")
    print("=" * 70)
    from approach_2_nlp_bow.recommender import NlpBagOfWordsRecommender

    bow_recommender = NlpBagOfWordsRecommender(MOVIES)
    bow_index = bow_recommender.fit()
    print(
        f"Built {len(bow_index.titles)}x{len(bow_index.titles)} matrix "
        f"in {bow_index.build_seconds:.4f}s\n"
    )
    for query in QUERIES:
        print(f"  {query}:")
        for bow_rec in bow_recommender.recommend(query, top_n=5):
            print(f"    - {bow_rec.title} ({bow_rec.similarity_score:.3f})")
    print()

    print("=" * 70)
    print("Approach 3: TF-IDF Content-Based Filtering")
    print("=" * 70)
    from approach_3_tfidf_content_based.recommender import TfidfContentBasedRecommender

    tfidf_recommender = TfidfContentBasedRecommender(MOVIES)
    tfidf_index = tfidf_recommender.fit()
    print(
        f"Built {len(tfidf_index.titles)}x{len(tfidf_index.titles)} matrix "
        f"({tfidf_index.vocabulary_size} terms) in {tfidf_index.build_seconds:.4f}s\n"
    )
    for query in QUERIES:
        print(f"  {query}:")
        for tfidf_rec in tfidf_recommender.recommend(query, top_n=5):
            print(f"    - {tfidf_rec.title} ({tfidf_rec.similarity_score:.3f})")
    print()


if __name__ == "__main__":
    main()
