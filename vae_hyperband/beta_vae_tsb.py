# β-VAE Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

class BetaVAE(nn.Module):
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

# β-VAE loss function
def beta_vae_loss(recon, x, mu, logvar, beta=4.0):
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kld, bce, kld

# # Train model
# model = BetaVAE(latent_dim=32).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# num_epochs = 10
# beta = 1.0

# for epoch in range(num_epochs):
#     model.train()
#     total_loss, total_bce, total_kld = 0, 0, 0
#     for x, _ in train_loader:
#         x = x.to(device)
#         optimizer.zero_grad()
#         recon, mu, logvar = model(x)
#         loss, bce, kld = beta_vae_loss(recon, x, mu, logvar, beta)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         total_bce += bce.item()
#         total_kld += kld.item()
#     save_name = "checkpoints/beta_vae" + str(epoch+1) + ".pth"
#     torch.save(model.state_dict(), save_name)
#     print(f"Epoch {epoch+1}, Total Loss: {total_loss:.2f}, BCE: {total_bce:.2f}, KLD: {total_kld:.2f}")

# # Visualize results
# import numpy as np
# model.eval()
# with torch.no_grad():
#     x, _ = next(iter(train_loader))
#     x = x.to(device)[:8]
#     recon, _, _ = model(x)
#     comparison = torch.cat([x, recon])
#     grid = make_grid(comparison.cpu(), nrow=8)
#     plt.figure(figsize=(12, 3))
#     plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)).squeeze(), cmap="gray")
#     plt.axis('off')
#     plt.title("Original (top) vs Reconstructed (bottom) with β-VAE")
#     plt.show()