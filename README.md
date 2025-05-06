# VAE from Scratch for Fashion Image Generation

 
**Summary:** Learn a clean latent representation of Fashion‑MNIST with a **Variational Autoencoder (VAE)** (and its β‑VAE variant) and generate realistic fashion images.  
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
```


---
## TensorBoard
asdf
![TensorBoard curves](vae_hyperband/tensorboard1.png)
![TensorBoard curves](vae_hyperband/tensorboard2.png)

---
## Best_config
![TensorBoard curves](vae_hyperband/best_config.png)

---

## Scaled MLOps Pipeline (📦 data → 🧠 model → 💾 checkpoints)

| Stage | File / Module | What It Does |
|-------|---------------|--------------|
| **Data Loading** | `train_vae.py › get_dataloader()`<br>`model.ipynb › Data block` | *Single* source of truth for downloading Fashion‑MNIST, normalizing, and wrapping it in a PyTorch `DataLoader`; works on CPU & GPU. |
| **Training Loop** | `train_vae.py › train_epoch()` | Handles forward pass, reconstruction + KL loss, back‑prop, metric aggregation, and Ray Tune callbacks. |
| **Validation** | `train_vae.py › eval_epoch()` | Runs every `config["val_interval"]` epochs; logs recon loss & KL to TensorBoard. |
| **Checkpointing** | `train_vae.py › save_ckpt()`<br>Auto‑handled by **Ray Tune** | Saves `state_dict`, optimizer state, and epoch/step number in `checkpoints/`. Resume training via `--resume_ckpt path/to/file.pt`. |
| **Experiment Tracking** | TensorBoard (`runs/…`)<br>Ray Tune JSON logs | Scalars: recon_loss, KL_divergence, total_loss.<br>Images: input, reconstruction, random samples. |
| **Cluster Execution** | `slurm_job.sh` | Portable SLURM script—sets up Conda env, launches Ray Tune with the correct GPU/CPU allocation, and pipes logs to `logs/`. |

> **Why it matters:** these pieces turn a classroom demo into a **repeatable experiment pipeline**—you can stop/restart jobs, sweep hyper‑parameters at scale, and visualise progress in real time.


