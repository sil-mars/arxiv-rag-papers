from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class Generator:

  def __init__(self):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", force_download=False, low_cpu_mem_usage=True)

  def clean_citations(self, text, valid_nums):
    def replace(match):
        content = match.group(1).strip()
        if content.isdigit() and int(content) in valid_nums:
            return f"[{content}]"
        return ""
    return re.sub(r'\[([^\]]+)\]', replace, text)

  def generate(self, context, question, valid_nums, refs):
    prompt = f"""<|im_start|>system
    You are a scientific assistant that answers questions using only the provided paper excerpts.
    
    RULES:
    - Answer the question directly and in detail
    - Use information from ALL provided papers
    - Maximum 5 bullet points
    - Each bullet MUST end with a number citation like [1], [2], or [3]
    - Valid citation numbers: {valid_nums}
    - Use ONLY these numbers, no others
    - If information is not in the papers say: "Not found in the provided papers."
    
    FORMAT EXAMPLE:
    * Transformers dominate NLP tasks [1]
    * They outperform RNNs on long-range dependencies [2]
    <|im_end|>
    
    <|im_start|>user
    Context:
    {context}
    
    Question:
    {question}
    <|im_end|>
    
    <|im_start|>assistant
    *"""

    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    outputs = self.model.generate(
        **inputs,
        max_new_tokens=250,
        eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=self.tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.2
    )

    input_length = inputs["input_ids"].shape[1]
    result = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    result = result.split("I don't know")[0].strip()
    result = self.clean_citations(result, valid_nums)

    # filter non-cited bullet points
    lines = result.split("\n")
    filtered = [l for l in lines if not l.strip().startswith("*") or re.search(r'\[\d+\]', l)]
    result = "\n".join(filtered)

    # filter unused citations
    used_nums = set(int(m) for m in re.findall(r'\[(\d+)\]', result))
    filtered_refs = "\n".join([
        line for line in refs.split("\n")
        if any(f"[{n}]" in line for n in used_nums)
    ])

    return result + f"\n\nReferences:\n{filtered_refs}"