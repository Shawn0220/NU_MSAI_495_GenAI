# 📧 Explainable GPT-2 for Email Subject Line Generation

This project fine-tunes a GPT-2 model to generate subject lines from email bodies using the [AESLC](https://huggingface.co/datasets/Yale-LILY/aeslc) dataset. It further pursues **Explainable AI** by visualizing both attention and attribution to help understand model behavior.

---

## 🚀 Project Goals

- **Input:** Realistic email bodies (from AESLC dataset)
- **Output:** A concise subject line summarizing the email
- **Explainability:** Use both *Integrated Gradients* and *Attention Visualization* to interpret predictions

---

## 🧠 Model Training

- **Base model:** `gpt2` from HuggingFace Transformers
- **Training objective:** causal language modeling (`email_body + "\nSubject:" → subject_line`)
- **Data:** 12,000 samples from AESLC for training
- **Fine-tuning:** Done using Huggingface's `Trainer`

Example format:
Question (input): Let's schedule a meeting to discuss the roadmap.\nSubject:
Answer (target): Meeting Agenda


---

## 📊 Decoding Strategies Evaluation

We compare different decoding strategies to evaluate the quality and diversity of generated subject lines:

- `Greedy decoding`
- `Top-k sampling (k=50)`
- `Top-p (nucleus) sampling (p=0.9)`

A test script compares the model's generations against the ground-truth titles on the last 100 samples of the dataset, and saves results in `decoding_comparison.csv`.

---

## 🔍 Explainability Module

### ✅ Integrated Gradients

- Highlights which input tokens most influenced a particular output token.
- Generates token-to-token **attribution heatmaps**.

### ✅ Attention Visualization

- Extracts GPT-2’s final-layer self-attention for each generated token.
- Visualizes which parts of the email body were attended to when generating each token.

### ✅ Dual Heatmap Output

- Integrated Gradients + Attention Side-by-side
- Example:
  ![example](explain_meeting_agenda_dual.png)
Given the input "Let us schedule a meeting to discuss the Q2 roadmap and budget forecast.\nSubject:", the trained model generated "Agenda eting me" instead of the expected output "Meeting Agenda".
This suggests the model has learned relevant vocabulary but struggles with word ordering, indicating a potential weakness in sequence modeling or decoding strategy.
---

### ✅ Different decoding strategies Example
- Input: This email is acknowledgement from the Power Pool of Alberta of the change in direct sales/forward contract registration for contracts that are currently registered with the Power Pool between Suncor Energy and Enron Canada Power Corporation for trading after December 29 HE 1 2001. On January 4 the Power Pool received your acknowledgement of Enron's change in the source asset for the direct sales/forward contract registrations for contract # 1421 to modify the source from the Sundance 3 unit (SD3) to Enron's unmetered source asset (ECP-). The modified registration terms will be effective December 29 HE 1 2001 and will apply until the expiry date of the contract registration for net settlement purposes. If you have any questions relating to this matter, please give me a call.
- Ground Truth: Power Pool
-  Greedy: Power Pool Change
-  Top-k: Enron - changes in central authority registration
-  Top-p: Power Pool Rev. Registration 

## 📁 Key Files

| File | Description |
|------|-------------|
| `train_gpt2.py` | GPT2 training on AESLC |
| `evaluate_decoding_strategies.py` | Run inference on 100 samples with multiple decoding strategies |
| `explain_dual_heatmap.py` | Generate dual heatmap of attention and attribution for a given sample |
| `decoding_comparison.csv` | Output file with decoding results |
| `explain_meeting_agenda_dual.png` | Example heatmap visualization |

---

