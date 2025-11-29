"""Multi-vector–only RAG ablation: load index, retrieve with multi-vector model,
generate LLM answers, and save predictions."""
import json
import os
import torch  # Added for SentenceTransformer and device handling
import faiss  # Added for index search
import numpy as np  # Added for array handling
import pickle  # Added to load the index file
from datasets import load_dataset

# Assuming the main module exports these global variables and functions
from main import (
    load_llm,
    load_embedding_models,  # KEEP import but DO NOT CALL it in generate_multi_predictions
    build_prompt,
    generate_answer,
    extract_final_answer,
    # These global state variables MUST be exported by main.py to be modified here:
    INDEX_DATA,
    MULTI_MODEL
)
from sentence_transformers import SentenceTransformer  # Added for embedding generation

# --- Configuration ---
VALIDATION_DATASET = "izhx/COMP5423-25Fall-HQ-small"
# Prediction output file
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "test_predict_multi.jsonl")
# Ablation index path
ABLATION_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "index", "ablation_multi_index.pkl")

TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ablation_index():
    """
    Loads the specific ablation index file and populates the global INDEX_DATA from main.
    This replaces the call to load_hybrid_indices() which lacked the index_path argument.
    """
    global INDEX_DATA

    if not os.path.exists(ABLATION_INDEX_PATH):
        print(f"ERROR: Ablation index file not found at {ABLATION_INDEX_PATH}")
        return

    print(f"Loading ablation index from: {ABLATION_INDEX_PATH}")
    try:
        with open(ABLATION_INDEX_PATH, 'rb') as f:
            # We load the ablation data into the INDEX_DATA global variable from main.py
            INDEX_DATA = pickle.load(f)
        print("Ablation index loaded successfully.")
    except Exception as e:
        print(f"ERROR loading index file: {e}")


def load_multi_vector_model_local():
    """
    Loads the Multi-vector model locally, avoiding the problematic load_embedding_models in main.py.
    """
    global MULTI_MODEL
    global INDEX_DATA

    if MULTI_MODEL is not None:
        return

    model_name = INDEX_DATA.get('multi_model_name')

    if model_name:
        print(f"Loading Multi-vector Encoder model (Local): {model_name}")
        MULTI_MODEL = SentenceTransformer(model_name, device=DEVICE)
    else:
        print("ERROR: Multi-vector model name is None in ablation index.")


def multi_retrieve(query, k=10):
    """
    Retrieves documents using ONLY the Multi-vector (E5) Faiss index.
    This function implements the retrieval logic locally based on the globals provided by main.py.
    """
    global INDEX_DATA
    global MULTI_MODEL  # Accessing the global model loaded by load_multi_vector_model_local()

    if INDEX_DATA is None or 'multi_faiss_index' not in INDEX_DATA:
        print("ERROR: Multi-vector index not loaded. Check load_ablation_index.")
        return []

    # 1. Check Model Availability
    if MULTI_MODEL is None:
        print("ERROR: Multi-vector model not available. Check load_multi_vector_model_local().")
        return []

    # 2. Prepare and encode the query
    # E5 models typically require 'query: ' prefix
    prefixed_query = f"query: {query}"

    # Get query embedding using the loaded MULTI_MODEL
    query_vector = MULTI_MODEL.encode(
        prefixed_query,
        convert_to_tensor=True
    ).cpu().numpy().reshape(1, -1).astype('float32')

    # 3. Search the Faiss index
    multi_faiss_index = INDEX_DATA['multi_faiss_index']
    scores, indices = multi_faiss_index.search(query_vector, k)

    # 4. Format results
    results = []
    doc_ids = INDEX_DATA['doc_ids']
    docs = INDEX_DATA['docs']

    for score, index in zip(scores[0], indices[0]):
        if index >= 0 and index < len(doc_ids):
            # Docs contains the full text
            results.append({
                "id": doc_ids[index],
                "score": float(score),
                "text": docs[index]['text']
            })

    return results


def generate_multi_predictions():
    print("--- Starting Multi-Vector Ablation Prediction ---")
    print("Loading everything... (LLM, embeddings, ablation index)")

    # 1. Load LLM
    load_llm()

    # 2. Load Ablation Index (Populates global INDEX_DATA)
    load_ablation_index()

    # 3. Load Multi-vector model (Local loading to avoid main.py's problematic function)
    load_multi_vector_model_local()

    print("Loading dataset...")
    dataset = load_dataset(VALIDATION_DATASET)["validation"]

    results = []

    print(f"Total questions: {len(dataset)}")
    for i, item in enumerate(dataset):
        qid = item["id"]
        question = item["text"]

        # Use Multi-vector index only for retrieval
        retrieved = multi_retrieve(question, k=TOP_K)

        # Build prompt using the top 5 documents
        prompt = build_prompt(question, retrieved[:5])

        # Get LLM output
        raw_output = generate_answer(prompt)

        # Extract final answer
        answer = extract_final_answer(raw_output)

        # Store results and document info
        doc_list = [[d["id"], d["score"]] for d in retrieved]

        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "retrieved_docs": doc_list
        })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")

    print("Saving to jsonl:", OUT_PATH)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done. Output file: test_predict_multi.jsonl")


if __name__ == "__main__":
    generate_multi_predictions()