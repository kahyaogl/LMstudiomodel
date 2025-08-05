from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np
import requests
import json
import os
from pdf_cache import save_cache, load_cache
import faiss


model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # <-- model burada tanımlanıyor
def read_pdf_paragraphs(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    paragraphs = [p.strip() for p in text.split('.') if len(p.strip()) > 50]
    return paragraphs


def embed_paragraphs(paragraphs):
    return model.encode(paragraphs)


embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def find_best_match_faiss(paragraphs, paragraph_embeddings, question):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    paragraph_embeddings = np.array(paragraph_embeddings).astype("float32")

    # Normalize (opsiyonel, cosine similarity için)
    faiss.normalize_L2(paragraph_embeddings)
    faiss.normalize_L2(question_embedding)

    index = faiss.IndexFlatIP(paragraph_embeddings.shape[1])  # İç çarpım (cosine benzeri)
    index.add(paragraph_embeddings)

    D, I = index.search(question_embedding, 1)
    best_index = I[0][0]
    similarity = D[0][0]

    return paragraphs[best_index], similarity


def ask_local_llm(context, question):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "messages": [
            {
                "role": "user",
                "content": f"Aşağıdaki bağlamı dikkate alarak soruyu yanıtla:\n\nBağlam:\n{context}\n\nSoru:\n{question}"
            }
        ],
        "temperature": 0.3,
        "max_token" :300,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()['choices'][0]['message']['content']

def ask_question_from_pdf(pdf_path, question):
    paragraphs, paragraph_embeddings = load_cache(pdf_path)
    if paragraphs is None:
        paragraphs = read_pdf_paragraphs(pdf_path)
        paragraph_embeddings = embed_paragraphs(paragraphs)
        save_cache(pdf_path, paragraphs, paragraph_embeddings)
    context, sim = find_best_match_faiss(paragraphs, paragraph_embeddings, question)
    answer = ask_local_llm(context, question)
    return answer




