from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class E5Embedder:
    _instance = None

    def __init__(self):
        self.model = SentenceTransformer("intfloat/e5-base-v2")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        # Cosine similarity
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1, emb2)

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        # Returns cosine similarity matrix
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.dot(normed, normed.T)

def embed_text(text: str) -> np.ndarray:
    return E5Embedder.get_instance().embed_text(text)

def embed_texts(texts: List[str]) -> np.ndarray:
    return E5Embedder.get_instance().embed_texts(texts)

def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return E5Embedder.get_instance().similarity(emb1, emb2)

def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return E5Embedder.get_instance().similarity_matrix(embeddings) 