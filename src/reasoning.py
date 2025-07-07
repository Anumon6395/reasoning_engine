import numpy as np
from src.embedding import embed_text
from src.search import search_similar_by_embedding

def gauss_seidel_chain(query_text: str, max_iter: int = 10, tol: float = 1e-3):
    chain = []
    current_embedding = embed_text(query_text)
    prev_diff_norm = None
    for step in range(max_iter):
        # Search for the closest problem
        results = search_similar_by_embedding(current_embedding, top_k=1)
        if not results:
            break
        closest = results[0]
        # Get embedding of closest problem
        closest_embedding = embed_text(closest['text'])
        diff = current_embedding - closest_embedding
        diff_norm = np.linalg.norm(diff)
        if diff_norm < tol:
            break
        if prev_diff_norm is not None and diff_norm > prev_diff_norm:
            break
        chain.append({
            'step': step + 1,
            'index': closest['index'],
            'distance': closest['distance'],
            'file': closest['file'],
            'text': closest['text'],
            'diff_norm': float(diff_norm)
        })
        current_embedding = diff
        prev_diff_norm = diff_norm
    return chain 