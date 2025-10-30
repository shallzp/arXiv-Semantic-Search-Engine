# Semantic Search Engine for Scientific Papers 🔬

## Overview
This project builds a search engine for scientific research papers that compares traditional keyword search with modern semantic search techniques. The system enables users to enter natural language queries and retrieve the most relevant papers from a large corpus using two distinct approaches:
- **Keyword Search:** Uses TF-IDF vectorization and cosine similarity to find papers matching query keywords exactly.
- **Semantic Search:** Uses Sentence-BERT embeddings to find conceptually similar papers based on the meaning of abstracts.

## Dataset
The project uses the [arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download), a repository of scholarly articles across many scientific disciplines. Approximately 1 million papers are used from the full dataset of ~2.8 million.

Key columns used:
- `id`
- `authors`
- `title`
- `abstract`
- and metadata fields for filtering and display

## Methodology

### Data Preprocessing
- Text cleaning includes lowercasing, punctuation removal, and stopword filtering to prepare abstracts for TF-IDF vectorization.

### Keyword Search (TF-IDF)
- TF-IDF vectorizer built with `max_features=500000`, ignoring very frequent (`max_df=0.8`) and very rare (`min_df=5`) terms.
- Abstracts converted to sparse TF-IDF vectors.
- User queries are cleaned and vectorized the same way.
- Cosine similarity between query and document vectors ranks documents for retrieval.

### Semantic Search (Sentence-BERT)
- Pre-trained `all-MiniLM-L6-v2` Sentence-BERT model encodes all abstracts into 384-dimensional dense vectors.
- Queries are encoded similarly to produce embeddings.
- Cosine similarity between query and document embeddings ranks papers based on semantic relevance.

## How to Use
- Run the Streamlit app (`app.py`) to enter queries.
- View side-by-side results from keyword and semantic searches.
- Explore full paper abstracts and metadata for the top results.

## Future Work
- Extend to hybrid search combining keyword and semantic scores.
- Improve indexing and retrieval speed for larger datasets.
- Add relevance feedback and user personalization features.

## Acknowledgments
- Dataset provided by Cornell University’s arXiv.
- Sentence-BERT model from the Hugging Face community.

---

## Code Snippets

### TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=500000, max_df=0.8, min_df=5)
tfidf_matrix = vectorizer.fit_transform(df['abstract_clean'])

### Keyword Search
def tfidf_search(query, vectorizer, tfidf_matrix, df, top_k=5):
query_clean = clean_text(query)
query_vec = vectorizer.transform([query_clean])
cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
top_indices = cosine_sim.argsort()[-top_k:][::-1]
return df.iloc[top_indices]


### Semantic Search with Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['abstract'].tolist(), batch_size=64)

def semantic_search(query, model, embeddings, df, top_k=5):
query_emb = model.encode([query])
scores = cosine_similarity(query_emb, embeddings)
top_indices = scores.argsort()[-top_k:][::-1]
return df.iloc[top_indices]

---

This project demonstrates key differences in retrieval results between traditional keyword and modern semantic search methods, providing an educational foundation for building enhanced scientific search engines.