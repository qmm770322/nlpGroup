import json
import os
import pickle
import numpy as np
from datasets import load_dataset
from main import (
    load_llm,
    build_prompt,
    generate_answer,
    extract_final_answer
)
from rank_bm25 import BM25Okapi  # 确保 BM25Okapi 可用
import string
from nltk.tokenize import word_tokenize

# --- Configuration ---
VALIDATION_DATASET = "izhx/COMP5423-25Fall-HQ-small"
# *** 指向 BM25 专用的索引文件 ***
INDEX_PATH_BM25 = os.path.join(os.path.dirname(__file__), "..", "index", "bm25_index.pkl")
# 输出文件，用于保存 BM25 Only 模式的预测结果
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "test_predict_bm25.jsonl")

TOP_K = 10  # 检索 Top K 文档数量 (用于评估检索指标)
PASS_K = 5  # 传递给 LLM 的文档数量

# --- BM25 专用加载和检索逻辑 ---
INDEX_DATA_BM25 = None


def preprocess_text(text):
    """Lowercase, remove punctuation and tokenize text (Duplicated from main/builder)."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)


def load_bm25_index_data():
    """加载 BM25 检索所需的索引数据 (bm25_index.pkl)。"""
    global INDEX_DATA_BM25
    if INDEX_DATA_BM25 is not None:
        return INDEX_DATA_BM25['bm25'], INDEX_DATA_BM25['doc_ids'], INDEX_DATA_BM25['docs']

    if not os.path.exists(INDEX_PATH_BM25):
        raise FileNotFoundError(f"BM25 Index file '{INDEX_PATH_BM25}' not found. Run the BM25 builder script first.")

    print(f"Loading BM25 index data from {INDEX_PATH_BM25}...")
    with open(INDEX_PATH_BM25, 'rb') as f:
        INDEX_DATA_BM25 = pickle.load(f)
    print("BM25 Index loaded.")
    return INDEX_DATA_BM25['bm25'], INDEX_DATA_BM25['doc_ids'], INDEX_DATA_BM25['docs']


def retrieve_bm25_only(query: str, k: int = 10) -> list[dict]:
    """
    [BM25 Only Retrieval] 执行纯粹的 BM25 检索。
    返回的格式与 hybrid_retrieve 相同。
    """
    # 只需要加载 BM25 数据
    bm25, doc_ids, docs = load_bm25_index_data()

    # 获取 BM25 结果索引和分数
    tokenized_query = preprocess_text(query)
    doc_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(doc_scores)[::-1][:k]

    # 构造结果
    id_to_doc = {doc['id']: doc['text'] for doc in docs}
    final_results = []

    for i in top_k_indices:
        doc_id = doc_ids[i]
        final_results.append({
            'id': doc_id,
            'score': float(doc_scores[i]),
            'text': id_to_doc.get(doc_id, "ERROR: Document not found")
        })

    return final_results


def generate_predictions():
    """
    运行 BM25 Only 模式的预测生成。
    """
    print("--- BM25 Only Evaluation Started ---")

    # 1. 加载 LLM (使用 main.py 中的加载函数)
    load_llm()
    # 2. 加载 BM25 索引数据 (使用本脚本的专用加载函数)
    load_bm25_index_data()

    print("Loading dataset...")
    # NOTE: 这里不调用 load_embedding_models 或 load_hybrid_indices
    dataset = load_dataset(VALIDATION_DATASET)["validation"]

    results = []

    print(f"Total questions: {len(dataset)}")
    for i, item in enumerate(dataset):
        qid = item["id"]
        question = item["text"]

        # *** 检索步骤替换为 BM25 ONLY ***
        retrieved = retrieve_bm25_only(question, k=TOP_K)

        # 3. 构建 Prompt (使用 main.py 中的函数)
        # 注意：这里传递给 Prompt 的文档数量为 PASS_K (5个)，用于与原系统对齐
        prompt = build_prompt(question, retrieved[:PASS_K])

        # 4. 获取 LLM 输出 (使用 main.py 中的函数)
        raw_output = generate_answer(prompt)

        # 5. 提取答案 (使用 main.py 中的函数)
        answer = extract_final_answer(raw_output)

        # 存储结果 (存储 TOP_K 的检索文档，但只用 PASS_K 的文档生成)
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

    print("Done. Output saved to test_predict_bm25.jsonl. Use eval scripts to compare against hybrid.")


if __name__ == "__main__":
    generate_predictions()