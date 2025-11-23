import streamlit as st
import time
# 导入您的后端函数
from main import (
    load_llm,
    load_hybrid_indices,
    load_embedding_models,
    run_rag_system,
    add_to_history,
    CONVERSATION_HISTORY
)

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="COMP5423 RAG System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 COMP5423 Group Project: RAG Chatbot")
st.markdown("Supports **Multi-Turn (Feature A)** & **Agentic Workflow (Feature B)**")


@st.cache_resource
def init_system():
    # 纯加载逻辑
    load_llm()
    load_hybrid_indices()
    load_embedding_models()
    return True


# 在外部处理 UI 反馈
st.toast("🚀 正在加载 LLM、索引和嵌入模型 (仅首次运行)...", icon="⏳")

with st.spinner("Loading Models & Indices... (This may take a minute)"):
    # 在 spinner 内部调用缓存函数
    init_system()


# --- 3. 初始化聊天历史 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. 显示历史消息 (默认折叠日志和文档) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 仅当它是带元数据的 System Answer 时，才显示详细信息
        if message["role"] == "assistant" and "logs" in message:

            # 历史记录默认折叠 (expanded=False)
            with st.expander("🕵️ Agentic Workflow Logs (Reasoning & Verification)", expanded=False):
                # 历史日志，简单显示
                for log in message["logs"]:
                    st.info(log)

        if message["role"] == "assistant" and "docs" in message:
            # 历史记录默认折叠 (expanded=False)
            with st.expander("📚 Retrieved Evidence (Source Documents)", expanded=False):
                for i, doc in enumerate(message["docs"]):
                    # 历史记录中也显示 RRF Score
                    score = doc.get('score', 0.0)
                    st.markdown(f"**Doc {i + 1}** (ID: `{doc['id']}`) **| RRF Score:** `{score:.4f}`")
                    st.caption(doc['text'])
                    st.divider()

# --- 5. 处理用户输入 ---
if prompt := st.chat_input("Ask a question (e.g., Where was Obama born?)..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示助手正在思考
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Thinking & Retrieving..."):
            try:
                # 调用后端 RAG 系统
                retrieved_docs, answer, effective_query, logs = run_rag_system(
                    prompt,
                    k=5,
                    is_multi_turn=True
                )

                # 更新全局历史 (Feature A)
                add_to_history(effective_query, answer)

                # --- 1. 代理工作流日志 (Feature B) ---
                # 当前回合默认展开 logs (expanded=True)
                with st.expander("🕵️ Agentic Workflow Logs (CoT 思考 & 验证)", expanded=True):
                    thinking_logs = [log for log in logs if 'Chain of Thought' in log or '🧠' in log]
                    verification_logs = [log for log in logs if 'Verification' in log or '🛡️' in log or '✅' in log]

                    if thinking_logs:
                        st.subheader("🧠 Chain of Thought (思考过程)")
                        # 使用 st.code 展示思考过程，格式清晰
                        st.code('\n'.join(thinking_logs), language='markdown')

                    if verification_logs:
                        st.subheader("🛡️ Self-Verification (自验证)")
                        for log in verification_logs:
                            # 根据结果使用不同颜色
                            if 'Passed' in log or '✅' in log:
                                st.success(log)
                            elif 'Failed' in log or 'Warning' in log:
                                st.warning(log)
                            else:
                                st.info(log)

                # --- 2. 检索结果 (Hybrid Retrieval) ---
                # 默认折叠文档，除非用户想看
                with st.expander("📚 检索到的源文档 (Hybrid RRF Score)", expanded=False):
                    for i, doc in enumerate(retrieved_docs):
                        score = doc.get('score', 0.0)
                        st.markdown(f"**Doc {i + 1}** (ID: `{doc['id']}`) **| RRF Score:** `{score:.4f}`")
                        st.caption(doc['text'])
                        st.divider()

                # --- 3. 展示最终答案 ---
                message_placeholder.markdown(answer)

                # 将完整交互保存到 session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "logs": logs,  # 保存日志以便历史回看
                    "docs": retrieved_docs  # 保存文档以便历史回看
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")

        # 强制重新运行以更新界面
        st.rerun()