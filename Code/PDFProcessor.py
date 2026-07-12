import arxiv
import fitz
import os
import requests
import numpy as np
import faiss

class PDFProcessor:

    def __init__(self, dirpath="pdfs/"):
        self.client = arxiv.Client()
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)

    def download(self, paper_id):
        path = os.path.join(self.dirpath, f"{paper_id}.pdf")
        if os.path.exists(path):
            return path
        
        url = f"https://arxiv.org/pdf/{paper_id}"
        response = requests.get(url)
        
        with open(path, "wb") as f:
            f.write(response.content)
        
        print(f"Downloaded: {paper_id}")
        return path

    def parse(self, paper_id):
        path = os.path.join(self.dirpath, f"{paper_id}.pdf")
        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text

    def chunk(self, text, words_lim=300, overlap=50):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+words_lim])
            chunks.append(chunk)
            i += words_lim - overlap 
        return chunks

    def download_and_parse(self, paper_id):
        self.download(paper_id)
        text = self.parse(paper_id)
        return self.chunk(text) 

    def retrieve_chunks(self, query, chunks, embedder, k=3):
        # Embed query
        query_emb = embedder.embed([query])
        
        # Embed chunks
        chunk_embs = embedder.embed(chunks)
        
        # Cosine similarity
        chunk_embs = chunk_embs.astype(np.float32)
        query_emb = query_emb.astype(np.float32)
        faiss.normalize_L2(chunk_embs)
        faiss.normalize_L2(query_emb)
        
        scores = query_emb @ chunk_embs.T  # scale product
        top_indices = scores[0].argsort()[::-1][:k]  # top k indexes
        
        return [chunks[i] for i in top_indices]