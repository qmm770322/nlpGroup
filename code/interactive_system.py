"""Interactive terminal UI for the RAG system with multi-turn and agentic modes."""
from main import (
    load_llm,
    load_hybrid_indices,
    load_embedding_models,
    add_to_history,
    CONVERSATION_HISTORY,
    extract_final_answer,
    run_rag_system
)

# 定义全局模式
MODE_MULTI_TURN = 'A'
MODE_AGENTIC = 'B'


def start_interactive_chat():
    print("--- RAG 交互式终端 (集成 Feature A & B) ---")
    load_llm()
    load_hybrid_indices()
    load_embedding_models()

    print("\n系统已就绪。支持多轮对话(A)和自动验证(B)。输入 'exit' 退出。")

    turn_count = 0
    while True:
        user_input = input(f"\n[Turn {turn_count + 1}] User: ").strip()
        if user_input.lower() == 'exit': break
        if not user_input: continue

        try:
            # 调用统一的函数
            # 注意：run_rag_system 内部会自动处理 history 重构(A) 和 verify(B)
            retrieved, answer, effective_query = run_rag_system(
                user_input,
                k=10,
                is_multi_turn=True
            )

            # 更新历史 (Feature A Requirement)
            add_to_history(effective_query, answer)

            print(f"\n[Turn {turn_count + 1}] System Answer: {answer}")

            # 打印部分检索结果作为证据展示
            if retrieved:
                print(f"| 支持证据 (Top 3):")
                for i, doc in enumerate(retrieved[:3]):
                    print(f"| - [{i + 1}] {doc['text'][:60]}...")

            turn_count += 1

        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    # 确保 NLTK punkt 资源已下载
    import nltk

    try:
        nltk.download('punkt', quiet=True)
    except LookupError:
        pass

    start_interactive_chat()