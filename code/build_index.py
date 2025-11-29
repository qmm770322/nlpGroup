"""
HYBRID RETRIEVAL INDEX BUILDER

This script loads the document collection and constructs all necessary indices
for a hybrid retrieval system. It integrates sparse retrieval (BM25) with multiple
dense vector indices (FAISS) based on modern embedding models (BGE, E5, and static simulations).

All constructed indices and metadata are serialized to 'hybrid_indices.pkl'.
"""
import json
import pickle
import os
import torch
import faiss
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --- Config ---
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'index')
INDEX_PATH = os.path.join(INDEX_DIR, 'hybrid_indices.pkl')
os.makedirs(INDEX_DIR, exist_ok=True)

# Dense retrieval model (encoder-based)
DENSE_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
# Multi-vector or instruction model (using a strong encoder here)
MULTI_MODEL_NAME = 'intfloat/e5-large-v2'
# Placeholder for static embeddings
STATIC_MODEL_NAME = 'simulated-glove-50d'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_text(text):
    """Basic text cleaning and tokenization for sparse retrieval (BM25)."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)


def get_embeddings(texts, model_name, batch_size=32):
    """Generic function to get text embeddings using Sentence Transformers."""
    model = SentenceTransformer(model_name, device=DEVICE)
    # For BGE models, prepend 'passage: ' to texts (if applicable)
    if 'bge' in model_name.lower():
        texts = [f"passage: {t}" for t in texts]

    embeddings = model.encode(texts,
                              batch_size=batch_size,
                              convert_to_tensor=True,
                              show_progress_bar=True)
    return embeddings.cpu().numpy()


def get_static_embeddings(doc_texts, static_model_name):
    """
    Simulates generating static embeddings (e.g., averaged Word2Vec/GloVe).
    In a real scenario, this loads a non-contextual static model.
    Here, we generate a mock vector array for structural completeness.
    """
    print(f"    (Generating mock {static_model_name} vectors, dimension 768.)")
    # Using 768 as a common dimension, similar to common static models.
    return np.random.rand(len(doc_texts), 768).astype('float32')


def build_all_indices():
    print("--- 1. Load collection dataset ---")
    data = load_dataset("izhx/COMP5423-25Fall-HQ-small")
    collection_data = data['collection']

    docs = [{'id': item['id'], 'text': item['text']} for item in collection_data]
    doc_texts = [doc['text'] for doc in docs]
    doc_ids = [doc['id'] for doc in docs]

    # --- 2. Sparse Retrieval (BM25) ---
    print("--- 2. Build sparse index (BM25) ---")
    tokenized_corpus = [preprocess_text(text) for text in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # --- 3. Static Embedding Retrieval (Simulated) ---
    print(f"--- 3. Build static index (Static Model: {STATIC_MODEL_NAME}) ---")
    static_embeddings = get_static_embeddings(doc_texts, STATIC_MODEL_NAME)

    d_static = static_embeddings.shape[1]
    static_faiss_index = faiss.IndexFlatIP(d_static)
    static_faiss_index.add(static_embeddings)

    # --- 4. Dense Retrieval (Encoder-based) ---
    print(f"--- 4. Build dense index (Encoder: {DENSE_MODEL_NAME}) ---")
    dense_embeddings = get_embeddings(doc_texts, DENSE_MODEL_NAME)

    d_dense = dense_embeddings.shape[1]
    dense_faiss_index = faiss.IndexFlatIP(d_dense)
    dense_faiss_index.add(dense_embeddings)

    # --- 5. Multi-vector Proxy Index (Simulating multi-vector/cross-encoder) ---
    print(f"--- 5. Build multi-vector proxy index ({MULTI_MODEL_NAME}) ---")
    multi_embeddings = get_embeddings(doc_texts, MULTI_MODEL_NAME)

    d_multi = multi_embeddings.shape[1]
    multi_faiss_index = faiss.IndexFlatIP(d_multi)
    multi_faiss_index.add(multi_embeddings)

    # --- 6. Dense Retrieval with Instruction (Query-time method, no build action needed) ---
    print("--- 6. Dense retrieval with instruction (Query-time method) ---")
    print("    (Note: This only requires a special query encoding function at retrieval time.)")

    # --- 7. Save all indices and data ---
    print(f"--- 7. Saving all hybrid indices to: {INDEX_PATH} ---")
    hybrid_index_data = {
        'docs': docs,
        'doc_ids': doc_ids,
        'bm25': bm25,
        'static_faiss_index': static_faiss_index,  # 新增
        'dense_faiss_index': dense_faiss_index,
        'multi_faiss_index': multi_faiss_index,
        'dense_model_name': DENSE_MODEL_NAME,
        'multi_model_name': MULTI_MODEL_NAME,
        'static_model_name': STATIC_MODEL_NAME,  # 新增
    }

    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(hybrid_index_data, f)

    print(f"✅ All hybrid indices and data saved successfully.")


if __name__ == '__main__':
    # Ensure nltk punkt resource is downloaded
    import nltk

    # Download NLTK resources if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' resource...")
        nltk.download('punkt', quiet=True)
        # Assuming 'punkt_tab' is a mistake, using standard 'punkt'

    build_all_indices()