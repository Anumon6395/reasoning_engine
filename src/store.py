# Placeholder for storage logic 

import os
import json
import numpy as np

DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'db')
PROBLEMS_PATH = os.path.join(DB_DIR, 'problems.json')
EMBEDDINGS_DIR = os.path.join(DB_DIR, 'embeddings')

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def load_problem_metadata():
    if not os.path.exists(PROBLEMS_PATH):
        return {}
    with open(PROBLEMS_PATH, 'r') as f:
        return json.load(f)

def save_problem_metadata(metadata):
    # Sort by integer index
    sorted_metadata = {str(k): v for k, v in sorted(((int(k), v) for k, v in metadata.items()))}
    with open(PROBLEMS_PATH, 'w') as f:
        json.dump(sorted_metadata, f, indent=2)

def save_embedding(problem_id, vector):
    path = os.path.join(EMBEDDINGS_DIR, f'{problem_id}.npy')
    np.save(path, vector)

def load_embedding(problem_id):
    path = os.path.join(EMBEDDINGS_DIR, f'{problem_id}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding not found for id {problem_id}")
    return np.load(path)

def check_problem_exists(problem_hash):
    metadata = load_problem_metadata()
    for meta in metadata.values():
        if meta.get('hash') == problem_hash:
            return True
    return False

def generate_id(text):
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest() 