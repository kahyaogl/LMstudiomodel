# LMmodel.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np
import requests
import json
import os
from pdf_cache import save_cache, load_cache


model = SentenceTransformer("all-MiniLM-L6-v2")

def read_pdf_paragraphs(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    paragraphs = [p.strip() for p in text.split('.') if len(p.strip()) > 50]
    return paragraphs
# LMmodel.py içinde veya uygun bir modülde
from sentence_transformers import SentenceTransformer

def embed_paragraphs(paragraphs):
    return  model.encode(paragraphs)


def find_best_match(paragraphs, paragraph_embeddings, question):
    question_embedding = model.encode([question])
    sims = cosine_similarity(question_embedding, paragraph_embeddings)[0]
    best_idx = np.argmax(sims)

    combined_context = paragraphs[best_idx]
    if best_idx + 1 < len(paragraphs):
        combined_context += " " + paragraphs[best_idx + 1]
    return combined_context, sims[best_idx]

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
        "temperature": 0.7,
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
    context, sim = find_best_match(paragraphs, paragraph_embeddings, question)
    answer = ask_local_llm(context, question)
    return answer



