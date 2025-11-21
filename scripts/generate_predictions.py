import json
import os
import re
from datasets import load_dataset
from main import (
    load_llm,
    load_embedding_models,
    load_hybrid_indices,
    hybrid_retrieve,
    build_prompt,
    generate_answer,
    extract_final_answer
)

# dataset and output path
VALIDATION_DATASET = "izhx/COMP5423-25Fall-HQ-small"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "test_predict.jsonl")

TOP_K = 10





def generate_predictions():
    print("Loading everything... (LLM, embeddings, index)")
    load_llm()
    load_hybrid_indices()
    load_embedding_models()

    print("Loading dataset...")
    dataset = load_dataset(VALIDATION_DATASET)["validation"]

    results = []

    print(f"Total questions: {len(dataset)}")
    for i, item in enumerate(dataset):
        qid = item["id"]
        question = item["text"]

        # retrieval
        retrieved = hybrid_retrieve(question, k=TOP_K)

        # build prompt with top documents
        prompt = build_prompt(question, retrieved[:5])

        # get LLM output
        raw_output = generate_answer(prompt)

        # extract clean final answer
        answer = extract_final_answer(raw_output)

        # store id + answer + doc info
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

    print("Done.")


if __name__ == "__main__":
    generate_predictions()
