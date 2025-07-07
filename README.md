# Reasoning Engine

A modular, extensible engine for embedding, storing, and reasoning over problems using vector search and iterative algorithms.

## Features
- Embed problems using state-of-the-art language models
- Store embeddings and metadata with deduplication
- Search for similar problems using FAISS
- Iterative reasoning with Gauss-Seidel-like logic
- Clean storage abstraction for easy maintenance
- Batch and single problem support
- CLI tools for all major workflows

## Project Structure
```
reasoning_engine/
  db/
    index.faiss         # FAISS index for fast similarity search
    problems.json       # Metadata for all embedded problems
    embeddings/         # Individual .npy files for each problem's embedding
  scripts/
    embed_problem.py    # CLI for embedding and managing problems
    search_similar.py   # CLI for searching similar problems
    gauss_seidel_chain.py # CLI for iterative reasoning
  src/
    embedding.py        # Embedding logic
    search.py           # Search logic
    reasoning.py        # Gauss-Seidel and other reasoning logic
    store.py            # Storage abstraction
    ...
```

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **(Optional) Download model weights in advance** for faster first run.

## Usage

### 1. Embed Problems
#### Single Problem
```bash
python -m scripts.embed_problem --file path/to/problem.txt
```
- Embeds the problem, adds it to the FAISS index and metadata, and saves the embedding as a .npy file.
- Skips duplicates by content hash.

#### Batch Embedding
```bash
python -m scripts.embed_problem --batch path/to/batch_file.txt
```
- Embeds all problems (one per line) in the file, skipping duplicates.

#### List All Problems
```bash
python -m scripts.embed_problem --list
```
- Lists all embedded problems with their index and a text preview.

#### Remove a Problem
```bash
python -m scripts.embed_problem --remove INDEX
```
- Removes the problem with the given index from the database, metadata, and embeddings.

---

### 2. Search for Similar Problems
```bash
python -m scripts.search_similar --file path/to/query.txt --topk 3
python -m scripts.search_similar --text "What is the capital of France?" --topk 5
```
- Searches for the top-k most similar problems in the database to the query (from file or string).
- Prints the results with index, distance, file, and text.

---

### 3. Gauss-Seidel Chain Reasoning
```bash
python -m scripts.gauss_seidel_chain --file path/to/query.txt --max-iter 10 --tol 1e-3
python -m scripts.gauss_seidel_chain --text "Your query here" --max-iter 5 --tol 0.01
```
- Iteratively searches for the closest problem, computes the difference vector, and repeats until convergence or divergence.
- Prints the chain of problems traversed, with step info, index, distance, diff norm, file, and text.

---

## Storage Details
- **problems.json:** Maps integer indices to problem metadata (file, text, hash).
- **embeddings/:** Each problem's embedding is stored as a .npy file named by its index.
- **index.faiss:** The FAISS index for fast similarity search.

## Extending the Engine
- Add new reasoning algorithms in `src/reasoning.py`.
- Add new search strategies in `src/search.py`.
- Use `src/store.py` for all storage and metadata operations.

## Contributing
- Please add tests for new features.
- Keep code modular and DRY.

---

For questions or suggestions, open an issue or contact the maintainer. 