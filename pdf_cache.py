import pickle
import os

def save_cache(pdf_path, paragraphs, embeddings):
    base = os.path.basename(pdf_path)
    name = os.path.splitext(base)[0]
    with open(f'cache/{name}.pkl', 'wb') as f:
        pickle.dump((paragraphs, embeddings), f)

def load_cache(pdf_path):
    base = os.path.basename(pdf_path)
    name = os.path.splitext(base)[0]
    path = f'cache/{name}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None, None