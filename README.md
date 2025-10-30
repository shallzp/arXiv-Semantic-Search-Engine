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




This project demonstrates key differences in retrieval results between traditional keyword and modern semantic search methods, providing an educational foundation for building enhanced scientific search engines.