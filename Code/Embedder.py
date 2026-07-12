import numpy as np
from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    
    def embed(self, texts, batch_size=32): 
        if isinstance(texts[0], dict):
            texts = [d["text"] for d in texts]
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    def save(self, emb, path):
        np.save(path, emb)
    
    def load(self, path):
        return np.load(path)