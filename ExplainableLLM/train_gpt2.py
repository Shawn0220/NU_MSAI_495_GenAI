from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset, Dataset
import torch
from transformers import Trainer, TrainingArguments

ds = load_dataset("Yale-LILY/aeslc", split="train")

data = [{"question": item["email_body"] + "\nSubject:", "answer": item["subject_line"]} for item in ds]

print(data[:3])


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # 避免 pad token 问题

# data = [{"question": "What's the capital of Japan?", "answer": "Tokyo."}]

def preprocess(example):
    prompt = f"Q: {example['question']}\nA:"
    full_text = prompt + " " + example["answer"]

    tokenized = tokenizer(
        full_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors=None,   # 要的是 list[int] 而不是 tensor
    )

    labels = tokenized["input_ids"].copy()

    prompt_len = len(tokenizer(prompt, truncation=True, max_length=128)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    tokenized["labels"] = labels
    return tokenized


dataset = Dataset.from_list(data[:12000]).map(preprocess, remove_columns=["question", "answer"])


model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer)) 

training_args = TrainingArguments(
    output_dir="./gpt2-qa-slurm",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    warmup_steps=20,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,               # 如果使用支持的 GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
