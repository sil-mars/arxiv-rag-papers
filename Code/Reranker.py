from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, docs):
        # pairs (query, doc)
        pairs = [(query, d["text"]) for d in docs]

        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)

        # sort score + doc
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        return [(doc, float(score)) for score, doc in ranked]