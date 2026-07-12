import os
import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = ""

with open("eval_results.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ChatOpenAI(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings()
)

print(results)
results.to_pandas().to_csv("eval_scores.csv", index=False)
print("Saved eval_scores.csv")
