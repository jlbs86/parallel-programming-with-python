# Movie Recommendation System: 3 Approaches Compared

This project implements and compares three different movie recommendation
system approaches, each based on analyzing a real-world reference article:

| # | Approach | Reference article |
|---|---|---|
| 1 | Vector-based (embeddings + Qdrant ANN search) | [How to Create a Vector-Based Movie Recommendation System Using Qdrant DB](https://medium.com/@akriti.upadhyay/how-to-create-a-vector-based-movie-recommendation-system-using-qdrant-db-e026554f49ec) |
| 2 | NLP + Bag-of-Words (RAKE + CountVectorizer) | [Movie Recommendation Engine with NLP](https://www.analyticsvidhya.com/blog/2022/01/movie-recommendation-engine-with-nlp/) |
| 3 | TF-IDF content-based filtering | [Film Recommendation System Using Content-Based Filtering](https://medium.com/web-mining-is688-spring-2021/film-recommendation-system-using-content-based-filtering-eb856c7359f3) |

All three approaches run against the **same shared dataset**
(`data/movies.py`, 30 movies with title, director, actors, plot, and
genres) so their recommendations can be fairly compared.

## Project structure

```text
recommender_system_approaches/
├── data/
│   └── movies.py                      # shared movie catalog
├── approach_1_vector_qdrant/
│   └── recommender.py                 # sentence-transformers embeddings + Qdrant ANN search
├── approach_2_nlp_bow/
│   └── recommender.py                 # RAKE keyword extraction + Bag-of-Words + cosine similarity
├── approach_3_tfidf_content_based/
│   └── recommender.py                 # TF-IDF vectorization + cosine similarity
└── compare_all_approaches.py          # runs all three side by side
```

## How each approach works

### Approach 1: Vector-based (Qdrant)

Embeds each movie's description into a 384-dimensional semantic vector
using the `sentence-transformers/all-MiniLM-L6-v2` transformer model,
stores the vectors in an in-memory Qdrant collection (cosine distance,
HNSW index), and recommends by running a nearest-neighbor **vector
search** instead of a full pairwise comparison. This captures *semantic*
similarity -- it can recognize that "an astronaut stranded in space" and
"a crew lost in a wormhole" are thematically related, even without shared
vocabulary.

### Approach 2: NLP + Bag-of-Words

Extracts plot keywords with **RAKE** (Rapid Automatic Keyword
Extraction), merges actor/director names into single tokens to avoid
first-name collisions, concatenates genre + director + actors + keywords
into a "bag of words" per movie, vectorizes with raw word
**counts** (`CountVectorizer`), and computes pairwise **cosine
similarity** over the full catalog.

### Approach 3: TF-IDF content-based filtering

Combines plot text, genres, director, and actors into one text blob per
movie, vectorizes with **TF-IDF** (down-weighting common words, up-weighting
distinctive ones) instead of raw counts, and computes cosine similarity
(via the equivalent and faster `linear_kernel` on L2-normalized vectors).

## Key differences

| | Vectors | Weighting | Similarity search | Scales to millions of items? |
|---|---|---|---|---|
| Approach 1 | Dense semantic embeddings (384-dim) | Learned by transformer | Approximate NN (HNSW) | Yes |
| Approach 2 | Sparse lexical (vocabulary-sized) | Raw word counts | Exhaustive `n x n` matrix | No (O(n^2) memory) |
| Approach 3 | Sparse lexical (vocabulary-sized) | TF-IDF | Exhaustive `n x n` matrix | No (O(n^2) memory) |

## Running

Each approach is runnable standalone:

```bash
pip install scikit-learn rake-nltk nltk qdrant-client sentence-transformers
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"

python approach_1_vector_qdrant/recommender.py
python approach_2_nlp_bow/recommender.py
python approach_3_tfidf_content_based/recommender.py
```

Or run all three side by side:

```bash
python compare_all_approaches.py
```

## Sample output (30-movie catalog)

Querying `"The Godfather"`, all three approaches agree the #1
recommendation is `"The Godfather: Part II"`, but their confidence
differs sharply because of how they measure similarity:

- Approach 1 (semantic embeddings): **0.840** -- the transformer
  understands these are narratively and thematically the same saga.
- Approach 2 (Bag-of-Words counts): **0.300** -- driven mostly by the
  shared director and a few overlapping plot keywords.
- Approach 3 (TF-IDF): **0.193** -- lower than Approach 2 because TF-IDF
  specifically down-weights common words, leaving fewer distinctive
  shared terms between the two plots.

This illustrates the core trade-off: semantic (embedding-based) methods
generalize better across different wording, while lexical methods
(Bag-of-Words / TF-IDF) are cheaper, fully interpretable, and need no
external model, but only recognize similarity when the actual words
overlap.
