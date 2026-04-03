# 📘 Resource-Efficient RAG for Lok Sabha Question Answering

This repository presents a **Retrieval-Augmented Generation (RAG)** based system for answering questions from Indian **Lok Sabha parliamentary proceedings**. The system is designed to be **lightweight, CPU-efficient, and suitable for public-sector deployment**.



## 🚀 Overview

Parliamentary documents are large, unstructured, and difficult to query. This project transforms Lok Sabha PDFs into a **searchable and interactive question-answering system** using:

- Semantic retrieval (FAISS)
- Lightweight embeddings (MiniLM)
- Text generation (DistilGPT-2)



## ⚙️ System Architecture

1. **PDF Processing**
   - Extract text from Lok Sabha PDF documents

2. **Chunking**
   - Split text into overlapping chunks (300 tokens)

3. **Embedding**
   - Model: `all-MiniLM-L6-v2` (384-dimension vectors)

4. **Indexing**
   - FAISS (`IndexFlatIP`) with cosine similarity

5. **Retrieval**
   - Top-k retrieval (k = 3)

6. **Generation**
   - Model: DistilGPT-2


## 📂 Dataset

- Source: Official Lok Sabha repository  
- Period: **Monsoon Session 2023 → Budget Session 2024**
- Size: ~3,500 PDF documents  
- Content includes:
  - Debates  
  - Starred & Unstarred Questions  
  - Bill discussions  



## 🧪 Evaluation

The system is evaluated on **450 queries**, categorized as:

| Query Type | Description |
|-----------|------------|
| Simple    | Direct factual retrieval |
| Complex   | Contextual understanding |
| Compound  | Multi-step reasoning |



## 📊 Performance

| Metric | Value |
|------|------|
| Factual Accuracy | 94% |
| Relevance Score | 4.6 / 5 |
| Latency | ~1200 ms (CPU) |



## 🖥️ Installation

```bash
pip install -r requirements.txt
