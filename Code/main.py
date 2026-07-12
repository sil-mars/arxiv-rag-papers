from Loader import Loader
from Embedder import Embedding
from Indexer import Indexer
from Retriever import Retriever
from Generator import Generator
from Reranker import Reranker
from PDFProcessor import PDFProcessor
import os
import json
import torch
import gradio as gr

os.environ["HF_TOKEN"] = ""
path = "/net-vol/Data/arxiv-metadata-oai-snapshot.json"
artifacts_path = "/net-vol/artifacts/"
EVAL_MODE = False
os.makedirs(artifacts_path, exist_ok=True)

# 1. LOAD
loader = Loader(path)
data = loader.read_data()

# 2. EMBEDDINGS
embedder = Embedding()
emb_path = os.path.join(artifacts_path, "embedding.npy")
if os.path.exists(emb_path):
    emb = embedder.load(emb_path)
else:
    emb = embedder.embed(data)
    embedder.save(emb, emb_path)

# 3. INDEX
indexer = Indexer()
index_path = os.path.join(artifacts_path, "index.faiss")
if os.path.exists(index_path):
    indexer.load(index_path)
else:
    indexer.build_FAISS_flatcos(emb)
    indexer.save(index_path)

# 4. Reranker
reranker = Reranker()

# 5. PDF Processor
processor = PDFProcessor()

# 6. RETRIEVER + GENERATOR
retriever = Retriever(embedder, indexer, data)
gen = Generator()


def query(question):
    torch.cuda.empty_cache()
    docs, scores = retriever.retrieve(question, 50)
    reranked = reranker.rerank(question, docs)
    top = reranked[:3]

    paper_list = []
    id_to_num = {doc["id"]: i+1 for i, (doc, score) in enumerate(top)}

    for doc, score in top:
        num = id_to_num[doc["id"]]
        chunks = processor.download_and_parse(doc["id"])
        top_chunks = processor.retrieve_chunks(question, chunks, embedder, k=5)
        best = "\n".join([f"[{num}] {c}" for c in top_chunks])
        paper_list.append(best)

    context = "\n\n".join(paper_list)
    valid_nums = list(id_to_num.values())
    refs = "\n".join([f"[{id_to_num[doc['id']]}] {doc['title']} (arxiv:{doc['id']})" for doc, score in top])

    return gen.generate(context, question, valid_nums, refs)


def main():
    test_queries = [
        # cs.CL - NLP
        "What are transformers in NLP?",
        # cs.LG - GNN
        "What are graph neural networks used for?",
        # quant-ph - Quantum Computing
        "What are the main challenges in quantum computing?",
        # cs.CV - Computer Vision
        "How are transformers applied in computer vision?",
        # q-bio - Drug Discovery
        "How is machine learning applied to drug discovery?",
        # cs.RO - Robotics
        "What are the main approaches to reinforcement learning in robotics?",
        # cs.CR - Security
        "What are adversarial attacks in deep learning?",
        # cs.CE - Climate
        "How is deep learning used for climate modeling?",
        # eess.IV - Medical Imaging
        "How is AI used for medical image segmentation?",
        # cs.SE - Code Generation
        "What are the applications of large language models in code generation?",
        # cs.SD - Speech
        "How are neural networks used for speech recognition?",
        # cs.IR - Recommendation Systems
        "How do recommendation systems use collaborative filtering?",
        # cs.CY - Ethics / Bias
        "What are the main sources of bias in machine learning models?",
        # econ - Economics
        "How is machine learning used in financial markets and trading?",
        # astro-ph - Astronomy
        "How is deep learning used for galaxy classification and astronomical surveys?",
        # cond-mat - Materials Science
        "How is machine learning used to predict material properties?",
        # math - Mathematics
        "How are neural networks used to solve partial differential equations?",
        # stat - Statistics
        "What are the main approaches to Bayesian deep learning?",
        # q-fin - Quantitative Finance
        "How are recurrent neural networks applied to stock price prediction?",
        # eess.SP - Signal Processing
        "How is deep learning applied to EEG signal classification?",
    ]

    eval_results = []

    for q in test_queries:
        result = query(q)
        print("\nQUERY:", q)
        print(result)

        if EVAL_MODE:
            docs, scores = retriever.retrieve(q, 50)
            reranked = reranker.rerank(q, docs)
            top = reranked[:3]
            eval_results.append({
                "question": q,
                "answer": result,
                "contexts": [doc["text"] for doc, score in top]
            })

    if EVAL_MODE:
        with open("eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        print("\nSaved eval_results.json")


demo = gr.Interface(
    fn=query,
    inputs=gr.Textbox(label="Question", placeholder="Ask a question about any research topic..."),
    outputs=gr.Textbox(label="Answer"),
    title="ArXiv RAG System",
    description="Ask any research question and get answers grounded in arXiv papers."
)

if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    else:
        main()