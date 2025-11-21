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


# --- 2. 初始化系统 (缓存资源，避免重复加载) ---
@st.cache_resource
def init_system():
    load_llm()
    load_hybrid_indices()
    load_embedding_models()
    return True


with st.spinner("Loading Models & Indices... (This may take a minute)"):
    init_system()

# --- 3. 初始化聊天历史 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. 显示历史消息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果历史消息里包含证据或日志，也可以在这里渲染
        if "logs" in message:
            with st.expander("🕵️ Agentic Workflow Logs (Reasoning & Verification)"):
                for log in message["logs"]:
                    st.info(log)
        if "docs" in message:
            with st.expander("📚 Retrieved Evidence (Source Documents)"):
                for i, doc in enumerate(message["docs"]):
                    st.markdown(f"**Doc {i + 1} (ID: {doc['id']})**")
                    st.text(doc['text'])

# --- 5. 处理用户输入 ---
if prompt := st.chat_input("Ask a question (e.g., Where was Obama born?)..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示助手正在思考
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking & Retrieving..."):
            try:
                # 调用后端 RAG 系统
                # 注意：这里假设您已经按步骤2修改了 run_rag_system 以返回 logs
                retrieved_docs, answer, effective_query, logs = run_rag_system(
                    prompt,
                    k=5,
                    is_multi_turn=True
                )

                # 更新全局历史 (Feature A)
                add_to_history(effective_query, answer)

                # --- 展示 Bonus 内容 (Feature B) ---
                # 使用 Expander 折叠显示中间过程，保持界面整洁
                with st.expander("🕵️ Agentic Workflow Logs (Reasoning & Verification)", expanded=True):
                    for log in logs:
                        st.info(log)  # 蓝色信息框显示日志
                        time.sleep(0.1)  # 模拟处理过程的视觉效果

                # --- 展示检索结果 (Basic Requirement) ---
                # 显示检索到的文档
                with st.expander("📚 Retrieved Evidence (Source Documents)"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Doc {i + 1} (ID: {doc['id']})**")
                        st.caption(doc['text'])  # 使用 caption 显示较小的文本
                        st.divider()

                # --- 展示最终答案 ---
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