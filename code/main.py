"""Hybrid RAG core: load indices/models, perform hybrid retrieval, generate and verify LLM answers."""
import pickle
import os
import string
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# --- Paths and configuration ---
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'index', 'hybrid_indices.pkl')
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RRF_K = 60  # Reciprocal Rank Fusion parameter

model = None
tokenizer = None
DENSE_MODEL = None
MULTI_MODEL = None
INDEX_DATA = None
CONVERSATION_HISTORY = []


def add_to_history(question: str, answer: str):
    """Add QA pair to conversation history, keep last 3 rounds only."""
    global CONVERSATION_HISTORY
    if len(CONVERSATION_HISTORY) >= 6:
        CONVERSATION_HISTORY = CONVERSATION_HISTORY[2:]  # Remove oldest pair
    CONVERSATION_HISTORY.append({"role": "user", "content": question})
    CONVERSATION_HISTORY.append({"role": "assistant", "content": answer})


def preprocess_text(text):
    """Lowercase, remove punctuation and tokenize text."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)


def load_hybrid_indices():
    """Load hybrid retrieval indices from disk or cache."""
    global INDEX_DATA
    if INDEX_DATA is not None:
        return INDEX_DATA['bm25'], INDEX_DATA['doc_ids'], INDEX_DATA['docs'], INDEX_DATA['dense_faiss_index'], INDEX_DATA['multi_faiss_index']

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index file '{INDEX_PATH}' not found. Run build_all_indices.py first.")

    print(f"Loading indices from {INDEX_PATH}...")
    with open(INDEX_PATH, 'rb') as f:
        INDEX_DATA = pickle.load(f)
    print("Indices loaded.")
    return INDEX_DATA['bm25'], INDEX_DATA['doc_ids'], INDEX_DATA['docs'], INDEX_DATA['dense_faiss_index'], INDEX_DATA['multi_faiss_index']


def load_llm():
    """Load LLM model and tokenizer with 4-bit quantization."""
    global model, tokenizer
    if model is not None:
        return
    print(f"Loading LLM model '{LLM_MODEL_NAME}' on {DEVICE} with 4-bit quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("LLM loaded.")


def load_embedding_models():
    """Load dense and multi-vector embedding models."""
    global DENSE_MODEL, MULTI_MODEL, INDEX_DATA
    if DENSE_MODEL is None:
        print(f"Loading Dense Encoder model: {INDEX_DATA['dense_model_name']}")
        DENSE_MODEL = SentenceTransformer(INDEX_DATA['dense_model_name'], device=DEVICE)
    if MULTI_MODEL is None:
        print(f"Loading Multi-Vector Encoder model: {INDEX_DATA['multi_model_name']}")
        MULTI_MODEL = SentenceTransformer(INDEX_DATA['multi_model_name'], device=DEVICE)


def get_query_embedding(query: str, model: SentenceTransformer, model_key: str, is_instruction: bool = False):
    """Generate query embedding with optional instruction prefix."""
    global INDEX_DATA
    model_name = INDEX_DATA[model_key]
    if 'bge' in model_name.lower():
        query_text = f"query: {query}"
    elif is_instruction:
        query_text = f"Instruct: Retrieve documents that accurately answer the question. Query: {query}"
    else:
        query_text = query
    return model.encode(query_text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)


def retrieve_sparse(bm25, doc_ids, query, k=10):
    """Retrieve top-k documents using sparse BM25."""
    tokenized_query = preprocess_text(query)
    doc_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(doc_scores)[::-1][:k]
    return [doc_ids[i] for i in top_k_indices]


def retrieve_dense_encoder(faiss_index, doc_ids, query, k=10):
    """Retrieve top-k documents using dense embedding encoder."""
    if DENSE_MODEL is None:
        load_embedding_models()
    query_vector = get_query_embedding(query, DENSE_MODEL, 'dense_model_name', is_instruction=False)
    D, I = faiss_index.search(query_vector, k)
    return [doc_ids[i] for i in I[0]]


def retrieve_dense_instruction(faiss_index, doc_ids, query, k=10):
    """Retrieve top-k documents using dense instruction encoder."""
    if MULTI_MODEL is None:
        load_embedding_models()
    query_vector = get_query_embedding(query, MULTI_MODEL, 'multi_model_name', is_instruction=True)
    D, I = faiss_index.search(query_vector, k)
    return [doc_ids[i] for i in I[0]]


def retrieve_multi_vector_proxy(faiss_index, doc_ids, query, k=10):
    """Retrieve top-k documents using multi-vector proxy encoder."""
    if MULTI_MODEL is None:
        load_embedding_models()
    query_vector = get_query_embedding(query, MULTI_MODEL, 'multi_model_name', is_instruction=False)
    D, I = faiss_index.search(query_vector, k)
    return [doc_ids[i] for i in I[0]]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    fused_scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1.0 / (rank + k)
    sorted_fused_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_fused_docs]


def hybrid_retrieve(query, k=10):
    """Perform hybrid retrieval combining sparse and dense methods."""
    bm25, doc_ids, docs, dense_faiss, multi_faiss = load_hybrid_indices()
    sparse_list = retrieve_sparse(bm25, doc_ids, query, k=20)
    dense_encoder_list = retrieve_dense_encoder(dense_faiss, doc_ids, query, k=20)
    dense_instruction_list = retrieve_dense_instruction(multi_faiss, doc_ids, query, k=20)
    multi_vector_list = retrieve_multi_vector_proxy(multi_faiss, doc_ids, query, k=20)
    all_ranked_lists = [sparse_list, dense_encoder_list, dense_instruction_list, multi_vector_list]

    fused_doc_ids = reciprocal_rank_fusion(all_ranked_lists, k=RRF_K)
    id_to_doc = {doc['id']: doc['text'] for doc in docs}
    final_results = []
    for rank, doc_id in enumerate(fused_doc_ids[:k]):
        final_results.append({
            'id': doc_id,
            'score': RRF_K / (rank + RRF_K),
            'text': id_to_doc.get(doc_id, "ERROR: Document not found")
        })
    return final_results


def build_prompt(question, retrieved_docs):
    """
    Build ChatML prompt with Chain-of-Thought (CoT) instructions for Qwen.
    """
    context_list = []
    for i, doc in enumerate(retrieved_docs):
        context_list.append(f"Document [{i + 1}]: {doc['text']}")
    context_str = "\n---\n".join(context_list)

    system_instruction = (
        "You are a concise, multi-hop question answering system. "
        "Based **STRICTLY AND ONLY** on the provided documents. "
        "First, analyze the context and question step by step in a <think> block. "
        "Then, provide the final, concise answer in an <answer> block. "
        "If the documents DO NOT contain the answer, "
        "your <answer> block MUST say 'Information not found'. "
        "Your final output MUST contain both <think> and <answer> blocks, and nothing else."
    )

    user_message = (
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {question}"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def reformulate_query(current_question: str, history: list[dict]) -> str:
    """
    Rewrite follow-up query into a self-contained query using LLM and conversation history.

    Args:
        current_question: The user's latest, context-dependent query.
        history: The list of previous Q/A turns (CONVERSATION_HISTORY).

    Returns:
        The self-contained, reformulated query string.
    """
    global model, tokenizer, DEVICE  # 仍然需要访问 model, tokenizer, DEVICE

    messages = [
        {"role": "system",
         "content": "You are a query rewriter. Based on the CONVERSATION HISTORY and the FOLLOW-UP QUESTION, output only a single, self-contained search query. Do not output any other text or reasoning."}
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"CONVERSATION HISTORY ENDS.\n\nFOLLOW-UP QUESTION: {current_question}\n\nREFORMULATED SELF-CONTAINED QUERY:"
    })

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = generation_output[0, inputs.input_ids.shape[1]:]
    reformed_query = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return reformed_query.split('\n')[0].strip()

def generate_answer(prompt: str) -> str:
    """Generate answer from LLM with extended token limit for CoT."""
    global model, tokenizer, DEVICE
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = generation_output[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_rag_system(question: str, k: int = 5, is_multi_turn: bool = True) -> tuple[list[dict], str, str, list[str]]:
    """
    Integrated Feature A & B entry point.

    Flow:
    1. [Feature A] Reformulate Query (if history exists)
    2. [Retrieval] Hybrid Retrieve
    3. [Feature B] CoT Generation (Reasoning)
    4. [Feature B] Self-Verification

    Returns:
        retrieved_docs (list): List of document dictionaries.
        final_answer (str): The generated answer (verified).
        effective_query (str): The query actually used for retrieval.
        logs (list): A list of strings describing the agentic workflow steps for UI display.
    """
    retrieved_docs = []
    original_query = question
    effective_query = question
    logs = []  # 用于存储 Feature B 的中间日志，供 Web UI 展示

    # 1. Feature A: Multi-Turn Query Reformulation
    if is_multi_turn and len(CONVERSATION_HISTORY) > 0:
        print(f"-> [Feature A] Reformulating query based on history: '{question}'...")
        rewritten_query = reformulate_query(question, CONVERSATION_HISTORY)
        effective_query = rewritten_query

        print(f"-> [Feature A] Reformulated Query: '{effective_query}'")
        logs.append(f"🔄 [Feature A] Query Reformulated: {effective_query}")

    # 2. Retrieval
    retrieved_docs = hybrid_retrieve(effective_query, k=k)

    context_str_list = [doc['text'] for doc in retrieved_docs]
    context_str = "\n---\n".join(context_str_list)

    # 3. Feature B (Part 1): Reasoning & Generation
    raw_prompt = build_prompt(effective_query, retrieved_docs)
    raw_output = generate_answer(raw_prompt)

    initial_answer = extract_final_answer(raw_output)

    if "<think>" in raw_output:
        try:
            think_content = raw_output.split("<think>")[-1].split("</think>")[0].strip()
            print(f"-> [Feature B] Chain of Thought:\n{think_content}")
            logs.append(f"🧠 [Feature B] Chain of Thought:\n{think_content}")
        except IndexError:
            logs.append("🧠 [Feature B] Thinking trace found but parsing failed.")

    # 4. Feature B (Part 2): Self-Checking
    final_answer = initial_answer

    if initial_answer.lower() not in ["information not found.", "extraction failed."]:
        print("-> [Feature B] Verifying answer reliability...")
        logs.append(f"🛡️ [Feature B] Verifying answer reliability...")

        verified_result = verify_answer(effective_query, initial_answer, context_str)

        if verified_result.lower() == "information not found.":
            print(f"-> [Feature B] Warning: Initial answer '{initial_answer}' failed verification.")
            logs.append("Verification Failed: Evidence does not support the answer (Potential Hallucination).")
            final_answer = "Information not found (verified)."
        else:
            print("-> [Feature B] Verification passed.")
            logs.append("✅ Verification Passed: Answer is strictly supported by evidence.")
            final_answer = verified_result

    return retrieved_docs, final_answer, effective_query, logs

def extract_final_answer(text: str) -> str:
    """
    Try to get the clean answer from model output.
    Mainly look for <answer> ... and return the content inside.
    """

    t = text.lower()

    # check if model followed our format
    if "<answer>" in t:
        # split but keep original text
        part = text.split("<answer>", 1)[-1].strip()

        # if model added more tags then cut at next "<"
        if "<" in part:
            part = part.split("<")[0].strip()

        if part and part.lower() != "information not found":
            return part

    # if model directly says it could not find anything
    if "information not found" in t:
        return "Information not found."

    # fallback: return the last non-empty line
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        return lines[-1]

    return "Extraction Failed."

def verify_answer(question: str, initial_answer: str, context: str) -> str:
    """
    Feature B: 使用 LLM 检查生成的答案是否与检索到的上下文一致。
    返回验证后的答案或 'Information not found.'
    """
    global model, tokenizer, DEVICE

    VERIFICATION_SYSTEM_INSTRUCTION = (
        "You are a strict fact-checking agent. Your task is to verify if the 'Initial Answer' "
        "is strictly supported by the 'Context' for the EXACT SAME ENTITY asked in the 'Question'.\n"
        "CRITICAL RULES:\n"
        "1. Check for Entity Matching: If the question asks about 'Michelle Obama' but the document discusses 'Michelle DeYoung', you MUST reject it.\n"
        "2. No Assumptions: Do not infer relationships not stated in the text.\n"
        "3. Response Format: \n"
        "   - First, identifying the Subject in the Question and the Subject in the Evidence in a <think> block.\n"
        "   - Then, output <decision>SUPPORTED</decision> or <decision>NOT SUPPORTED</decision>.\n"
        "   - Finally, if NOT SUPPORTED, output 'Information not found'."
    )

    VERIFICATION_USER_MESSAGE = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        f"INITIAL ANSWER TO BE CHECKED: {initial_answer}"
    )

    messages = [
        {"role": "system", "content": VERIFICATION_SYSTEM_INSTRUCTION},
        {"role": "user", "content": VERIFICATION_USER_MESSAGE}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        verification_output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(verification_output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    if "<decision>" in output_text:
        decision_part = output_text.split("<decision>")[-1].strip()

        if "supported" in decision_part.lower():
            return initial_answer
        else:
            return "Information not found."

    return "Verification Error or Not Found."



if __name__ == '__main__':
    import nltk
    try:
        nltk.download('punkt', quiet=True)
    except LookupError:
        pass

    load_llm()
    load_hybrid_indices()
    load_embedding_models()

