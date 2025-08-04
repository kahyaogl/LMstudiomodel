from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np
import requests
import json


# === 1. PDF'den metin oku ===
def read_pdf_paragraphs(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    paragraphs = [p.strip() for p in text.split('.') if len(p.strip()) > 50]
    return paragraphs

# === 2. Embedding modeli yükle ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === 3. Paragrafları embed et ===
def embed_paragraphs(paragraphs):
    return model.encode(paragraphs)

# === 4. En uygun paragrafı bul ===
def find_best_match(paragraphs, paragraph_embeddings, question):
    question_embedding = model.encode([question])
    sims = cosine_similarity(question_embedding, paragraph_embeddings)[0]
    best_idx = np.argmax(sims)

    combined_context = paragraphs[best_idx]
    if best_idx + 1 < len(paragraphs):
        combined_context += " " + paragraphs[best_idx + 1]
    return combined_context, sims[best_idx]
import requests
import json

def ask_local_llm(context, question):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct-v0.3",  # Model ismi LM Studio'daki yüklü modele göre değiştirilmeli
        "messages": [
            {
                "role": "user",
                "content": f"Aşağıdaki bağlamı dikkate alarak soruyu yanıtla:\n\nBağlam:\n{context}\n\nSoru:\n{question}"
            }
        ],
        "temperature": 0.7,
        "stream": False
        # "max_tokens": 512  # Gerekirse eklenebilir
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response_json = response.json()
    return response_json['choices'][0]['message']['content']


# === 6. Hepsini birleştir ===
def ask_question_from_pdf(pdf_path, question):
    paragraphs = read_pdf_paragraphs(pdf_path)
    paragraph_embeddings = embed_paragraphs(paragraphs)
    context, sim = find_best_match(paragraphs, paragraph_embeddings, question)
    answer = ask_local_llm(context, question)
    return answer

# === 7. Örnek kullanım ===
if __name__ == "__main__":
    pdf_path = "pdf.pdf"  # PDF dosya adını buraya yaz
    question = "Front-End ve Güvenlik İlişkisi nedir?"
    cevap = ask_question_from_pdf(pdf_path, question)
    print("Cevap:\n", cevap)


