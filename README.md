# VAE from Scratch for Fashion Image Generation

![VAE](tensorboard2.png)

**Author & Course:** NU _MSAI 495 – Generative AI  
**Goal:** Learn a clean latent representation of Fashion‑MNIST with a **Variational Autoencoder (VAE)** (and its β‑VAE variant) and generate realistic fashion images.  
**Highlights**

- ✨ *From‑scratch* PyTorch implementation (no Lightning)  
- 🔍 **Hyperparameter search** with **Ray Tune** + ASHA/HyperBand (latent dim, LR, batch size)  
- 🏗️ **Scaled MLOps pipeline**: modular code, checkpoints, TensorBoard, SLURM‑ready  
- 📈 Reproducible results & best‑config plot  

---

## Dataset

> **Fashion‑MNIST** (Zalando Research) – 70 000 28×28 grayscale images across 10 clothing classes (T‑shirt, trouser, sneaker, …).  

---

## Model Architecture

| Part        | Layer(s) | Notes |
|-------------|----------|-------|
| **Encoder** | Conv (1→32, 3×3) → ReLU → Conv (32→64, 3×3) → ReLU → Flatten → FC → **μ**, **log σ²** | |
| **Latent**  | 32‑D vector sampled via reparameterization trick | Latent dim swept over {16, 32, 64} |
| **Decoder** | FC → reshape → ConvT (64→32) → ReLU → ConvT (32→1) → Sigmoid | |
| **Loss**    | **Binary Cross‑Entropy** + **KL‑divergence** | β‑VAE multiplies KL by β |

---

## Repository Structure

```text
vae_hyperband/
├── checkpoints/           # Auto‑saved model & Ray Tune checkpoints
├── logs/                  # TensorBoard event files
├── best_config.png        # Visualised best hyperparams
├── beta_vae_tsb.py        # Quick TB launch for β‑VAE runs
├── model.ipynb            # Notebook: VAE + β‑VAE definition & demo
├── slurm_job.sh           # Sample SLURM script (Ray on a cluster)
├── tensorboard1.png       # Loss / KL curves
├── tensorboard2.png       # Generated samples over epochs
└── train_vae.py           # Main entry – Ray Tune HPO + training loop
