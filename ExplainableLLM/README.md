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
- Input: I attach an updated list (to Feb 5) of the terminated contracts to date. F= or informational purposes, I have added a column that sets out the settleme= nt dates for contracts we have settled, so there may be some additional cou= nterparties added to my previous list where there was no ECC agreed termina= tion date but we have settled and terminated (no one needs to do anything f= or these c/ps). Again, the first table sets forth the physical contracts t= hat have been terminated and the second sets out the financial contracts, a= nd all data reflects ECC's (not the counterparties') perception of the worl= d. I have vetted my list against Blakes' master list and believe this is a= ccurate in all respects. As I indicated in my previous e-mails to you ther= e were a few date discrepancies and I have previously provided you with the= corrected dates for El Paso, Marathon, AEC Marketing, Calpine (Encal), Wil= liams Energy and Baytex (all physical), all which are now correctly reflect= ed in the attachment. A few others that were not previously on the list in= clude Coast Energy (physical) which auto terminated on Dec 2 and PrimeWest = Energy (physical) which auto terminated on Dec 2. =20 Thanks	Terminated Counterparty Contracts	Terminated Contracts	TW Contract Terminations	El Paso, Marathon, AEC Marketing, Cal
- Ground Truth Trader Meeting - 2/7 @ 7:45 a.m. (CST) - ECS 06752	Trader Meeting - 2/7 @ 7:45	Trader Meeting - 2/7 @ 7:45	Trader Meeting - 2/7 @ 7:45

## 📁 Key Files

| File | Description |
|------|-------------|
| `train_gpt2.py` | GPT2 training on AESLC |
| `evaluate_decoding_strategies.py` | Run inference on 100 samples with multiple decoding strategies |
| `explain_dual_heatmap.py` | Generate dual heatmap of attention and attribution for a given sample |
| `decoding_comparison.csv` | Output file with decoding results |
| `explain_meeting_agenda_dual.png` | Example heatmap visualization |

---

