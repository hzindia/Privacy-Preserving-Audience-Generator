import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(TabularVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Outputs values between 0 and 1
        )

    def reparameterize(self, mu, logvar, privacy_noise=1.0):
        """
        privacy_noise: 
        1.0 = Standard VAE behavior.
        >1.0 = More privacy (more noise), less utility.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * privacy_noise 
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # Reconstruction Loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld_loss
