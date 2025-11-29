import re
from main import (
    load_llm,
    load_embedding_models,
    load_hybrid_indices,
    hybrid_retrieve,
    build_prompt,
    generate_answer
)

TOP_K = 10


def extract_final_answer(text: str) -> str:
    """从 LLM 的结构化输出中，精确提取 <answer> 块的内容。"""

    # 将文本转换为小写以确保匹配，并查找 <answer> 标记
    lower_text = text.lower()

    if "<answer>" in lower_text:
        # 1. 以 <answer> 为分隔符进行分割
        # 使用原始文本进行分割，保留大小写
        answer_part = text.split("<answer>", 1)[-1].strip()

        # 2. 如果后面还有其他块（例如 </think> 或其他标签），则截断
        if "<" in answer_part:
            # 找到下一个尖括号开始的位置，并截断
            answer_part = answer_part.split("<")[0].strip()

        final_answer = answer_part.strip()

        # 确保答案不是空字符串
        if final_answer and final_answer.lower() != "information not found":
            return final_answer

    # 如果结构化提取失败，或者模型明确说了信息未找到，则返回标准错误信息
    if "information not found" in text.lower():
        return "Information not found."

    # 极端回退：如果模型输出了乱码或没有遵循结构
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else "Extraction Failed (No Answer Tag)."

def test_first_question():
    print("Loading models / indices ...")
    load_llm()
    load_hybrid_indices()
    load_embedding_models()

    print("Retrieving and generating answer for the first question...")

    # 只取验证集的第一个样本
    from datasets import load_dataset
    dataset = load_dataset("izhx/COMP5423-25Fall-HQ-small")["validation"]
    item = dataset[0]
    qid = item["id"]
    question = item["text"]

    retrieved = hybrid_retrieve(question, k=TOP_K)
    prompt = build_prompt(question, retrieved[:5])
    llm_output = generate_answer(prompt)
    answer = extract_final_answer(llm_output)

    print(f"Question ID: {qid}")
    print(f"Question: {question}")
    print("Retrieved Docs (top 5):")
    for doc in retrieved[:5]:
        print(f"- {doc['text']}")
    print("\nLLM Raw Output:\n", llm_output)
    print("\nExtracted Final Answer:", answer)

if __name__ == "__main__":
    test_first_question()
