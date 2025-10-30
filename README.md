# Semantic Search Engine for Scientific Papers 🔬


## Overview
This project builds a search engine for scientific research papers that compares traditional keyword-based search with modern semantic search techniques. The system enables users to enter natural language queries and retrieve the most relevant papers from a large corpus using two distinct approaches:
- **Keyword Search:** Uses TF-IDF vectorization paired with cosine similarity to find papers that match keywords in user queries exactly within abstracts and titles.
- **Semantic Search:** Uses Sentence-BERT embeddings to interpret the conceptual similarity between user queries and paper content, enabling retrieval based on the meaning of abstracts, titles, and optionally authors.


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
- Raw text data from paper abstracts, titles, and authors undergo cleaning:
   - Lowercase conversion
   - Punctuation removal
   - Whitespace trimming
   - For abstracts and titles: optional stopword removal and token normalization
- Author names are standardized for exact matching-based filter searches.

### Keyword Search (TF-IDF)
- Separate TF-IDF vectorizers are built for abstracts and titles with:
   - max_features=500000
   - ilter threshold parameters max_df=0.8 and min_df=5 to exclude noisy terms
- Abstracts and titles are vectorized into respective sparse matrices.
- User queries are pre-processed identically and transformed using the fitted vectorizer.
- Cosine similarity between query and document vectors ranks papers by keyword relevance.
- Author search uses exact substring matching over author name strings for filtering

### Semantic Search (Sentence-BERT)
- The pretrained SentenceTransformer model all-MiniLM-L6-v2 embeds abstracts, titles, and authors into a shared 384-dimensional semantic vector space.
- Embeddings capture contextual meaning beyond exact keyword matching.
- Query texts are encoded into embeddings, and cosine similarity calculates semantic closeness to each document embedding.
- This enables retrieval of papers conceptually matching the query even when words differ.

### Search Capabilities
- The system supports:
   - Content (abstracts) search via both TF-IDF keyword matching and Sentence-BERT semantic similarity
   - Title search with keyword and semantic methods
   - Author search with exact matching for high precision and semantic similarity optionally available
- Users can select search type and method via the UI to tailor matching granularity.

### Data and Model Persistence
- Fitted TF-IDF vectorizers and sparse matrices are serialized with pickle for reuse.
- Sentence-BERT embeddings are precomputed and stored for fast similarity queries.
- Efficient caching and loading mechanisms in the Streamlit app improve responsiveness.


## Quickstart

### Clone the Repository
```
git clone https://github.com/shallzp/arXiv-Semantic-Search-Engine.git
cd arXiv-Semantic-Search-Engine
```

### Install Dependencies
#### It's strongly recommended to use a virtual environment:
```
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Prepare Data
Download the arXiv dataset from https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download and place your data file inside the data/ directory.

### Run Notebooks
#### All steps for data cleaning, TF-IDF vectorization, and BERT embedding are provided in the notebooks/ folder:
- preprocessing.ipynb
- tf-idf.ipynb
- bert.ipynb

### Launch the Application
```
streamlit run app.py
```
Access the app via the link in your terminal (usually http://localhost:8501)


## Acknowledgments
- Dataset provided by Cornell University’s arXiv.
- Sentence-BERT model from the Hugging Face community.




This project demonstrates the key differences in retrieval results between traditional keyword-based search and modern semantic search methods, providing a comprehensive and educational foundation for building advanced scientific search engines that leverage both exact term matching and contextual understanding of paper content.