import os
import json
import numpy as np
import faiss
from src.embedding import embed_text

DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'db')
INDEX_PATH = os.path.join(DB_DIR, 'index.faiss')
PROBLEMS_PATH = os.path.join(DB_DIR, 'problems.json')


def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
    return faiss.read_index(INDEX_PATH)

def load_problems():
    if not os.path.exists(PROBLEMS_PATH):
        raise FileNotFoundError(f"Problems metadata not found at {PROBLEMS_PATH}")
    with open(PROBLEMS_PATH, 'r') as f:
        return json.load(f)

def search_similar(query_text: str, top_k: int = 1):
    index = load_faiss_index()
    problems = load_problems()
    query_emb = embed_text(query_text).astype(np.float32).reshape(1, -1)
    D, I = index.search(query_emb, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        meta = problems.get(str(idx), {})
        results.append({
            'index': idx,
            'distance': float(dist),
            'file': meta.get('file'),
            'text': meta.get('text'),
            'hash': meta.get('hash')
        })
    return results

def search_similar_by_embedding(query_emb: np.ndarray, top_k: int = 1):
    index = load_faiss_index()
    problems = load_problems()
    query_emb = query_emb.astype(np.float32).reshape(1, -1)
    D, I = index.search(query_emb, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        meta = problems.get(str(idx), {})
        results.append({
            'index': idx,
            'distance': float(dist),
            'file': meta.get('file'),
            'text': meta.get('text'),
            'hash': meta.get('hash')
        })
    return results 