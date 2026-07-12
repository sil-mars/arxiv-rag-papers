import faiss
import numpy as np

class Indexer:

    def __init__(self):
        self.index = None

    def build_FAISS_flatcos(self, embeddings):
        embeddings = embeddings.astype(np.float32)

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        print("Total vectors:", self.index.ntotal)

    def search(self, query_embedding, k=5): 
        query_embedding = query_embedding.astype(np.float32)

        faiss.normalize_L2(query_embedding)

        scores, indexes = self.index.search(query_embedding, k)

        return scores, indexes

    def save(self, path):
        faiss.write_index(self.index, path)
    
    def load(self, path):
        self.index = faiss.read_index(path)