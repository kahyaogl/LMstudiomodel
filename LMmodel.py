# LMmodel.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np
import requests
import json
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def read_pdf_paragraphs(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    paragraphs = [p.strip() for p in text.split('.') if len(p.strip()) > 50]
    return paragraphs

def get_cached_embeddings(pdf_path):
    os.makedirs("embeds", exist_ok=True)
    os.makedirs("paragraphs", exist_ok=True)

    base = os.path.basename(pdf_path).replace(".pdf", "")
    embed_path = f"embeds/{base}.npy"
    para_path = f"paragraphs/{base}.json"

    if os.path.exists(embed_path) and os.path.exists(para_path):
        embeddings = np.load(embed_path)
        with open(para_path, "r", encoding="utf-8") as f:
            paragraphs = json.load(f)
    else:
        paragraphs = read_pdf_paragraphs(pdf_path)
        embeddings = model.encode(paragraphs)
        np.save(embed_path, embeddings)
        with open(para_path, "w", encoding="utf-8") as f:
            json.dump(paragraphs, f, ensure_ascii=False, indent=2)
    
    return paragraphs, embeddings

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
    paragraphs, paragraph_embeddings = get_cached_embeddings(pdf_path)
    context, sim = find_best_match(paragraphs, paragraph_embeddings, question)
    return ask_local_llm(context, question)



