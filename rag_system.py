# ================================
# Lok Sabha RAG QA System
# ================================

import os
import fitz  # PyMuPDF
import time
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import faiss

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ================================
# 1. LOAD PDF DOCUMENTS
# ================================
def extract_text_from_pdf(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            doc = fitz.open(path)
            text = ""

            for page in doc:
                text += page.get_text()

            documents.append(text)

    return documents


# ================================
# 2. TEXT CHUNKING
# ================================
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ================================
# 3. BUILD INDEX
# ================================
def build_index(documents):
    all_chunks = []

    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    print(f"Total chunks: {len(all_chunks)}")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, model, all_chunks


# ================================
# 4. LOAD GENERATION MODEL
# ================================
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model


# ================================
# 5. RETRIEVAL FUNCTION
# ================================
def retrieve(query, embed_model, index, chunks, k=3):
    query_vec = embed_model.encode([query])
    query_vec = np.array(query_vec).astype('float32')

    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)
    results = [chunks[i] for i in indices[0]]

    return results


# ================================
# 6. GENERATE ANSWER (RAG)
# ================================
def generate_answer(query, embed_model, index, chunks, tokenizer, model):
    context_chunks = retrieve(query, embed_model, index, chunks)

    context = " ".join(context_chunks)

    prompt = f"""
You are a parliamentary assistant.
Answer ONLY based on the given context.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


# ================================
# 7. EVALUATION
# ================================
def evaluate(queries, q_type, embed_model, index, chunks, tokenizer, model):
    results = []

    for q in queries:
        start_time = time.time()

        answer = generate_answer(q, embed_model, index, chunks, tokenizer, model)

        latency = time.time() - start_time

        results.append({
            "query": q,
            "type": q_type,
            "answer": answer,
            "latency": latency
        })

    return results


# ================================
# 8. MAIN FUNCTION
# ================================
def main():

    # Path to your PDF dataset
    pdf_folder = "datasets"

    print("Loading PDFs...")
    documents = extract_text_from_pdf(pdf_folder)

    print("Building FAISS index...")
    index, embed_model, chunks = build_index(documents)

    print("Loading generator...")
    tokenizer, model = load_generator()

    # Sample queries (replace with your 150 each)
    simple_queries = [
        "What is the objective of the scheme?",
        "How many beneficiaries are covered?"
    ]

    complex_queries = [
        "How does the government justify the policy?"
    ]

    compound_queries = [
        "What are the provisions and who introduced it?"
    ]

    print("Evaluating...")
    results = []

    results.extend(evaluate(simple_queries, "simple", embed_model, index, chunks, tokenizer, model))
    results.extend(evaluate(complex_queries, "complex", embed_model, index, chunks, tokenizer, model))
    results.extend(evaluate(compound_queries, "compound", embed_model, index, chunks, tokenizer, model))

    df = pd.DataFrame(results)

    df.to_excel("evaluation_results.xlsx", index=False)

    print("Results saved to evaluation_results.xlsx")


# ================================
# RUN
# ================================
if __name__ == "__main__":
    main()
