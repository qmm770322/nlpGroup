# NLPGroup11
## 🤖 Advanced RAG Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) system based on the HotpotQA HQ-small dataset.  
It integrates a multi-channel **Hybrid Retrieval** pipeline (BM25, static embeddings, dense encoder embeddings, and multi-vector instruction-tuned embeddings) together with an **Agentic Workflow** to perform complex multi-hop reasoning and produce reliable, verifiable answers.

------------------------------------------------------------
📦 Project Structure
------------------------------------------------------------
```bash
project-root
├── evaluate
├── index
└── code
    └── data
```
------------------------------------------------------------
⚙️ Environment & Dependencies
------------------------------------------------------------

This project is built with PyTorch and the Hugging Face ecosystem.
An NVIDIA GPU with CUDA is strongly recommended for better performance.

To get started, first clone the repository:
```bash
git clone https://github.com/qmm770322/nlpGroup.git
```
Then install the required dependencies via pip:
```bash
pip install -r requirements.txt
```
------------------------------------------------------------
📥 Model Downloads
------------------------------------------------------------

All models are automatically downloaded from Hugging Face when first used.

 - LLM (Generator): Qwen/Qwen2.5-3B-Instruct  
 - Dense Retriever: BAAI/bge-large-en-v1.5  
 - Multi-Vector Retriever: intfloat/e5-large-v2  


------------------------------------------------------------
🚀 How to Run the Project (3 Steps)
------------------------------------------------------------

### Step 1 — Build the hybrid index
This script processes HotpotQA data and creates:
 - a BM25 sparse lexical index
 - a static embedding index (simulated non-contextual embeddings)
 - a dense FAISS index using contextual embeddings (BGE)
 - a multi-vector FAISS index using instruction-tuned embeddings (E5)
 - a combined hybrid index that integrates all the above for robust retrieval


Run:
```bash
python code/build_index.py
```
Output will be stored under:
```bash
index/hybrid_indices.pkl
```

### Step 2 — Generate predictions
Creates model predictions for evaluation:
```bash
python code/generate_predictions.py
```
Output file:
```bash
test_predict.jsonl
```

### Step 3 — Launch the Web UI (Feature A & B Demo)
Start the Streamlit interface:
```bash
streamlit run code/web.py
```
This demo includes:
- Multi-hop QA  
- Agentic workflow (retrieve → verify → synthesize)  
- Multi-turn interaction support  

