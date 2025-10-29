import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import string
import pickle

# ========== Page Configuration ==========
st.set_page_config(
    page_title="arXiv Semantic Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Custom CSS ==========
st.markdown("""
    <style>
    /* Result cards */
    .result-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-left: 4px solid #ff4b4b;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .paper-title {
        color: #262730;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .paper-authors {
        color: #808495;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    
    .paper-abstract {
        color: #31333f;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }
    
    .paper-meta {
        color: #808495;
        font-size: 0.85rem;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    
    .paper-link {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.5rem 1.25rem;
        background: #ff4b4b;
        color: white !important;
        text-decoration: none;
        font-weight: 500;
        border-radius: 4px;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .paper-link:hover {
        background: #ff2b2b;
        transform: translateX(5px);
    }
    
    .category-badge {
        display: inline-block;
        background: #f0f2f6;
        color: #262730;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }
    
    .doi-link {
        color: #ff4b4b;
        text-decoration: none;
        font-weight: 500;
    }
    
    .doi-link:hover {
        text-decoration: underline;
    }
    
    /* Stats box */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .stat-box {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        flex: 1;
        min-width: 200px;
    }
    
    .stat-value {
        color: #ff4b4b;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .stat-label {
        color: #808495;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: #e6e9ef;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Utility functions ==========
def clean_text(text):
    """Preprocess user query to match TF-IDF cleaning."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = text.split()
    return ' '.join(words)

def truncate_text(text, max_length=None):
    """Return full text without truncation."""
    if pd.isna(text) or text == "":
        return "No abstract available"
    text = str(text)
    
    # If max_length is None or not provided, return full text
    if max_length is None:
        return text
    
    # Otherwise truncate if needed
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_authors(authors, max_authors=3):
    """Format authors list from comma-separated string."""
    if pd.isna(authors) or authors == "":
        return "Unknown authors"
    
    # Split by comma and clean up
    author_list = [a.strip() for a in str(authors).split(',')]
    
    if len(author_list) > max_authors:
        return ', '.join(author_list[:max_authors]) + f' <em>et al.</em> ({len(author_list)} total)'
    return ', '.join(author_list)

def format_categories(categories):
    """Format categories as badges."""
    if pd.isna(categories) or categories == "":
        return ""
    
    cats = [c.strip() for c in str(categories).split()]
    badges = ''.join([f'<span class="category-badge">{cat}</span>' for cat in cats[:3]])
    return badges

def format_date(date_str):
    """Format date string nicely."""
    if pd.isna(date_str) or date_str == "":
        return "N/A"
    return str(date_str)

def get_year_from_date(date_value):
    """Extract year from date value (handles strings, Timestamps, etc.)"""
    try:
        if pd.isna(date_value):
            return None
        if hasattr(date_value, 'year'):  # Timestamp object
            return str(date_value.year)
        date_str = str(date_value)
        # Try to extract year from various formats
        if len(date_str) >= 4:
            return date_str[:4]
        return None
    except:
        return None

# ========== Load data and models ==========
@st.cache_data
def load_data():
    df = pd.read_pickle('./data/processed_arxiv_data.pkl')
    return df

@st.cache_resource
def load_tfidf_model():
    with open('./data/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('./data/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return vectorizer, tfidf_matrix

@st.cache_resource
def load_bert_model_and_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.load('./data/abstract_embeddings.npy')
    return model, embeddings

# ========== Search Functions ==========
def tfidf_search(query, vectorizer, tfidf_matrix, df, top_k=5):
    query_clean = clean_text(query)
    query_vec = vectorizer.transform([query_clean])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices].copy().reset_index(drop=True)
    return results

def bert_search(query, model, embeddings, df, top_k=5):
    query_emb = model.encode([query])
    cosine_scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = cosine_scores.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices].copy().reset_index(drop=True)
    return results

def display_results(results):
    """Display search results in a clean card format."""
    if results.empty:
        st.warning("No results found.")
        return
    
    for idx, row in results.iterrows():
        meta_items = []
        date_parts = []

        if 'first_version_date' in row and not pd.isna(row['first_version_date']) and str(row['first_version_date']).strip():
            date_parts.append(f"Published: {format_date(row['first_version_date'])}")
        
        if 'update_date' in row and not pd.isna(row['update_date']) and str(row['update_date']).strip():
            date_parts.append(f"Updated: {format_date(row['update_date'])}")
        
        if date_parts:
            meta_items.append(f"📅 {' | '.join(date_parts)}")
        
        if 'num_versions' in row and not pd.isna(row['num_versions']):
            meta_items.append(f"📝 Version {int(row['num_versions'])}")
        
        if 'journal-ref' in row and not pd.isna(row['journal-ref']) and str(row['journal-ref']).strip():
            journal = str(row['journal-ref'])[:60]
            meta_items.append(f"📖 {journal}")
        else:
            meta_items.append("📖 ")
        
        if 'doi' in row and not pd.isna(row['doi']) and str(row['doi']).strip():
            meta_items.append(f'<a href="https://doi.org/{row["doi"]}" target="_blank" class="doi-link">🔗 DOI</a>')
        else:
            meta_items.append(f'<a href="https://arxiv.org/abs/{row["id"]}" target="_blank" class="doi-link">🔗 arXiv</a>')
        
        comments_section = ""
        if 'comments' in row and not pd.isna(row['comments']) and str(row['comments']).strip():
            comments_section = f'<div style="color: #808495; font-size: 0.85rem; margin-top: 0.5rem; font-style: italic;">💬 {str(row["comments"])}</div>'
        else:
            comments_section = '<div style="color: #808495; font-size: 0.85rem; margin-top: 0.5rem; font-style: italic;">💬 </div>'
        
        # Build the meta section
        meta_section = ""
        if meta_items:
            meta_items_html = ' • '.join(meta_items)
            meta_section = f'<div class="paper-meta">{meta_items_html}</div>'
        
        # Build the complete card HTML
        card_html = f"""
            <div class="result-card">
                <div class="paper-title">{idx + 1}. {str(row['title'])}</div>
                <div class="paper-authors">👥 {format_authors(row.get('authors', ''))}</div>
                <div style="margin-bottom: 0.75rem;">{format_categories(row.get('categories', ''))}</div>
                <div class="paper-abstract">{truncate_text(row.get('abstract', ''))}</div>
                {comments_section}
                {meta_section}
                <a href="https://arxiv.org/abs/{row['id']}" target="_blank" class="paper-link">🔗 View on arXiv →</a>
            </div>
        """
        
        with st.container():
            st.markdown(card_html, unsafe_allow_html=True)
            
# ========== Streamlit UI ==========
def main():
    st.title("🔬 arXiv Semantic Search Engine")
    st.markdown("Discover scientific papers using advanced search algorithms")

    # Load data and models
    with st.spinner("🔄 Loading models and data..."):
        df = load_data()
        vectorizer, tfidf_matrix = load_tfidf_model()
        model, embeddings = load_bert_model_and_embeddings()
    
    # Display statistics
    try:
        num_categories = df['primary_category'].nunique() if 'primary_category' in df.columns else 'N/A'
        
        st.markdown(f"""
            <div class="stats-container">
                <div class="stat-box">
                    <div class="stat-value">{len(df):,}</div>
                    <div class="stat-label">Total Papers</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{num_categories}</div>
                    <div class="stat-label">Categories</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
            <div class="stats-container">
                <div class="stat-box">
                    <div class="stat-value">{len(df):,}</div>
                    <div class="stat-label">Total Papers</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Create columns for search options
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., deep learning in medical imaging, quantum computing algorithms...",
            label_visibility="collapsed"
        )
    
    with col2:
        label_col, select_col = st.columns([1, 2])
        with label_col:
            st.markdown('<p style="margin-top: 0.5rem;">Results:</p>', unsafe_allow_html=True)
        with select_col:
            top_k = st.selectbox("Results", [5, 10, 15, 20], index=0, label_visibility="collapsed")
    
    with col3:
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Display results when query is entered
    if query and (search_button or query):
        # TF-IDF Results
        st.subheader("TF-IDF Results")
        st.caption("Keyword-based matching using term frequency")
        
        with st.spinner("Searching with TF-IDF..."):
            tfidf_res = tfidf_search(query, vectorizer, tfidf_matrix, df, top_k=top_k)
            display_results(tfidf_res, "TF-IDF")

        st.divider()

        # BERT Results
        st.subheader("BERT Results")
        st.caption("Semantic understanding using neural embeddings")
        
        with st.spinner("Searching with BERT..."):
            bert_res = bert_search(query, model, embeddings, df, top_k=top_k)
            display_results(bert_res, "BERT")

if __name__ == "__main__":
    main()