import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray import train
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("here load cuda")
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z).view(-1, 64, 7, 7)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

def train_vae(config):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    model = VAE(latent_dim=config["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(5):  # Keep short for tuning
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train.report({"loss": total_loss / len(train_loader.dataset)})
print("here defining funcs")
def main():
    print("here in main")
    config = {
        "latent_dim": tune.choice([16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128])
    }

    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=5, grace_period=1)
    print("here load scheduler")
    result = tune.run(
        train_vae,
        config=config,
        scheduler=scheduler,
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        num_samples=4,
        max_concurrent_trials=1,  # ✅ 限制并发数
        storage_path="file:///home/qhm7800/genai/vae_hyperband/logs",
        name="vae_hyperband"
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    best_config = best_trial.config
    with open("logs/best_config.txt", "w") as f:
        f.write(str(best_config))
    print("Best config:", best_config)

if __name__ == "__main__":
    main()
