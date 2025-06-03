import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 1. 加载模型和分词器 ===
model_path = "/home/qhm7800/MSAI337_NLP/HW1/gpt2-qa-slurm-align/checkpoint-9000"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path, output_attentions=True)
model.eval()

# === 2. 输入数据 ===
email_body = "Let us schedule a meeting to discuss the Q2 roadmap and budget forecast.\nSubject:"
subject_line = "Meeting Agenda"

# === 3. 编码输入和输出 ===
input_ids = tokenizer(email_body, return_tensors="pt").input_ids
target_ids = tokenizer(subject_line, return_tensors="pt").input_ids[0]

inputs_embeds = model.transformer.wte(input_ids)
baseline_embeds = torch.zeros_like(inputs_embeds)

# === 4. Integrated Gradients Attribution Matrix ===
ig = IntegratedGradients(lambda embeds: model(inputs_embeds=embeds).logits[:, -1, :])

# 添加 dummy token 便于生成第一个词
dummy = torch.zeros((1, 1, inputs_embeds.shape[-1]))
inputs_ext = torch.cat([inputs_embeds, dummy], dim=1)
baseline_ext = torch.cat([baseline_embeds, dummy.clone()], dim=1)

ig_matrix = []

for token_id in target_ids:
    attr, _ = ig.attribute(
        inputs=inputs_ext,
        baselines=baseline_ext,
        target=token_id.item(),
        return_convergence_delta=True
    )
    token_score = attr.sum(dim=-1).squeeze(0).detach().numpy()
    ig_matrix.append(token_score)

ig_matrix = np.stack(ig_matrix)  # shape: [target_len, input_len]

# === 5. Attention Matrix ===
# 手动逐 token 生成，并提取 attention
cur_input_ids = input_ids.clone()
attn_weights = []

for token_id in target_ids:
    outputs = model(
        input_ids=cur_input_ids,
        output_attentions=True,
        return_dict=True,
        use_cache=False
    )
    logits = outputs.logits
    pred_token = torch.argmax(logits[:, -1, :], dim=-1)

    # 提取最后层平均 attention
    attn = outputs.attentions[-1][0].mean(dim=0)
    attn_to_input = attn[-1, :input_ids.shape[1]].detach().numpy()
    attn_weights.append(attn_to_input)

    # 为了 attention 对齐，这里添加真实 token 而不是预测 token
    token_id = token_id.view(1, 1)  # 确保是 [1, 1] 形状
    cur_input_ids = torch.cat([cur_input_ids, token_id], dim=1)


attn_matrix = np.stack(attn_weights)

# === 6. 可视化并排输出 ===
input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
target_tokens = tokenizer.convert_ids_to_tokens(target_ids)

fig, axs = plt.subplots(1, 2, figsize=(min(36, len(input_tokens) * 1.2), 1.5 * len(target_tokens)))

# 左图：Integrated Gradients
sns.heatmap(
    ig_matrix,
    xticklabels=[t.replace("Ġ", " ") for t in input_tokens],
    yticklabels=[t.replace("Ġ", " ") for t in target_tokens],
    cmap="coolwarm", center=0,
    ax=axs[0]
)
axs[0].set_title("Integrated Gradients Attribution")
axs[0].set_xlabel("Email Body Tokens")
axs[0].set_ylabel("Generated Subject Tokens")

# 右图：Attention
sns.heatmap(
    attn_matrix,
    xticklabels=[t.replace("Ġ", " ") for t in input_tokens],
    yticklabels=[t.replace("Ġ", " ") for t in target_tokens],
    cmap="YlGnBu",
    ax=axs[1]
)
axs[1].set_title("Decoder Self-Attention (Last Layer)")
axs[1].set_xlabel("Email Body Tokens")
axs[1].set_ylabel("")

plt.tight_layout()
plt.savefig("explain_meeting_agenda_dual1.png", dpi=300)
plt.close()
print("✅ 可视化已保存为 explain_meeting_agenda_dual.png")
