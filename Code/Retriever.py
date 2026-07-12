import os
import numpy as np

class Retriever:

    def __init__(self, embedder, indexer, data):
        self.embedder = embedder
        self.indexer = indexer
        self.data = data

    def retrieve(self, question, k=50):
        query_emb = self.embedder.embed([question])
        scores, indexes = self.indexer.search(query_emb, k)
        
        seen_ids = {}
        for score, idx in zip(scores[0], indexes[0]):
            doc = self.data[idx]
            pid = doc["id"]
            if pid not in seen_ids or score > seen_ids[pid][0]:
                seen_ids[pid] = (score, doc)
        
        docs = [d for _, d in seen_ids.values()]
        return docs, scores