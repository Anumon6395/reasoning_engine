import argparse
import os
import json
import numpy as np
import faiss
import hashlib
import glob
from src.embedding import embed_text, embed_texts
from src import store

DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'db')
INDEX_PATH = os.path.join(DB_DIR, 'index.faiss')

EMBED_DIM = 768  # Update if GritLM-7B uses a different hidden size

def compute_text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def load_faiss_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    else:
        # Default to 768, but will be set dynamically when embedding is added
        return None

def save_faiss_index(index):
    faiss.write_index(index, INDEX_PATH)

def get_next_available_index(problems):
    used = set(int(k) for k in problems.keys())
    idx = 0
    while idx in used:
        idx += 1
    return idx

def embed_and_store(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    text_hash = compute_text_hash(text)
    problems = store.load_problem_metadata()
    for meta in problems.values():
        if meta.get('hash') == text_hash:
            print(f"Problem already embedded (hash match): {file_path}")
            return

    embedding = embed_text(text)
    embedding = embedding.astype(np.float32).reshape(1, -1)

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        assert embedding.shape[1] == index.d, f"Embedding dim {embedding.shape[1]} != index dim {index.d}"
    else:
        index = faiss.IndexFlatL2(embedding.shape[1])

    # Add to FAISS
    index.add(embedding)
    idx = get_next_available_index(problems)
    problems[str(idx)] = {
        'file': os.path.basename(file_path),
        'text': text[:200] + ('...' if len(text) > 200 else ''),
        'hash': text_hash
    }
    store.save_embedding(str(idx), embedding.squeeze())
    save_faiss_index(index)
    store.save_problem_metadata(problems)
    print(f"Embedded and stored: {file_path} as idx {idx}")

def embed_and_store_batch(batch_file_path):
    with open(batch_file_path, 'r') as f:
        problems_list = [line.strip() for line in f if line.strip()]
    existing = store.load_problem_metadata()
    new_problems = []
    new_hashes = []
    for text in problems_list:
        text_hash = compute_text_hash(text)
        if any(meta.get('hash') == text_hash for meta in existing.values()):
            print(f"Skipped duplicate: {text[:60]}...")
            continue
        new_problems.append(text)
        new_hashes.append(text_hash)
    if not new_problems:
        print("No new problems to embed.")
        return
    embeddings = embed_texts(new_problems).astype(np.float32)
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        assert embeddings.shape[1] == index.d, f"Embedding dim {embeddings.shape[1]} != index dim {index.d}"
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    # Assign the next available indices for each new problem
    for i, (text, text_hash) in enumerate(zip(new_problems, new_hashes)):
        idx = get_next_available_index(existing)
        existing[str(idx)] = {
            'file': f'batch_{idx}',
            'text': text[:200] + ('...' if len(text) > 200 else ''),
            'hash': text_hash
        }
        store.save_embedding(str(idx), embeddings[i])
    save_faiss_index(index)
    store.save_problem_metadata(existing)
    print(f"Embedded and stored {len(new_problems)} new problems.")

def list_problems():
    problems = store.load_problem_metadata()
    if not problems:
        print("No problems embedded yet.")
        return
    for idx, meta in problems.items():
        print(f"[{idx}] {meta['file']}: {meta['text']}")

def remove_problem(idx_to_remove):
    problems = store.load_problem_metadata()
    idx_to_remove = str(idx_to_remove)
    if idx_to_remove not in problems:
        print(f"No problem with index {idx_to_remove} found.")
        return
    del problems[idx_to_remove]
    # Remove embedding file if exists
    try:
        os.remove(os.path.join(DB_DIR, 'embeddings', f'{idx_to_remove}.npy'))
    except FileNotFoundError:
        pass
    # Rebuild FAISS index from remaining problems
    if problems:
        texts = [meta['text'] for meta in problems.values()]
        embeddings = embed_texts(texts).astype(np.float32)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        # Save all embeddings again
        for idx, emb in zip(problems.keys(), embeddings):
            store.save_embedding(str(idx), emb)
    else:
        index = faiss.IndexFlatL2(768)
    save_faiss_index(index)
    store.save_problem_metadata(problems)
    print(f"Removed problem with index {idx_to_remove}.")

def embed_directory(directory_path):
    """Embed all txt files in a directory, treating each file as a batch of problems."""
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in directory {directory_path}")
        return
    
    print(f"Found {len(txt_files)} .txt files in {directory_path}")
    
    for txt_file in txt_files:
        print(f"\nProcessing {os.path.basename(txt_file)}...")
        try:
            embed_and_store_batch(txt_file)
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Embed a problem and store in FAISS index.')
    parser.add_argument('--file', type=str, help='Path to problem text file to embed.')
    parser.add_argument('--list', action='store_true', help='List all embedded problems.')
    parser.add_argument('--remove', type=str, help='Remove problem by index.')
    parser.add_argument('--batch', type=str, help='Path to file with one problem per line.')
    parser.add_argument('--directory', type=str, help='Path to directory containing txt files to embed.')
    args = parser.parse_args()

    if args.remove:
        remove_problem(args.remove)
    elif args.directory:
        embed_directory(args.directory)
    elif args.batch:
        embed_and_store_batch(args.batch)
    elif args.list:
        list_problems()
    elif args.file:
        embed_and_store(args.file)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 