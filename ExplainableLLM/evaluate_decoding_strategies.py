import csv
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm

# === 1. 加载数据 ===
ds = load_dataset("Yale-LILY/aeslc", split="train")
data = [{"question": item["email_body"] + "\nSubject:", "answer": item["subject_line"]} for item in ds]
test_data = data[-20:]  # 后100条用于评估

# === 2. 加载模型和分词器 ===
model_path = "/home/qhm7800/MSAI337_NLP/HW1/gpt2-qa-slurm-align/checkpoint-9000"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # ✅ 加上这行

model = GPT2LMHeadModel.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))  # ✅ 再加这行


model.eval()

# === 3. 生成函数（不同策略） ===
@torch.no_grad()
def generate_subject(prompt, strategy="greedy"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if strategy == "greedy":
        output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
    elif strategy == "top-k":
        output = model.generate(input_ids, max_new_tokens=10, do_sample=True, top_k=50)
    elif strategy == "top-p":
        output = model.generate(input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
    else:
        raise ValueError("Unknown decoding strategy")

    gen = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return gen.strip()

# === 4. 遍历测试集并收集结果 ===
results = []
print("Running evaluation on 100 samples with 3 decoding strategies...")

for item in tqdm(test_data):
    prompt = item["question"]
    gt = item["answer"]

    greedy_out = generate_subject(prompt, "greedy")
    topk_out = generate_subject(prompt, "top-k")
    topp_out = generate_subject(prompt, "top-p")

    results.append({
        "email_body": prompt.replace("\nSubject:", ""),
        "ground_truth": gt,
        "greedy": greedy_out,
        "top-k": topk_out,
        "top-p": topp_out,
    })

# === 5. 保存为 CSV ===
csv_file = "decoding_comparison.csv"
with open(csv_file, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["email_body", "ground_truth", "greedy", "top-k", "top-p"])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ 保存完成：{csv_file}")
