import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. VAE Architecture (Standard MLP)
# ==========================================
class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(TabularVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar, privacy_noise=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * privacy_noise 
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

# ==========================================
# 2. GAN Architecture
# ==========================================
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. LSTM-VAE Architecture (New)
# ==========================================
class TabularLSTM_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(TabularLSTM_VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 32
        
        # Encoder: Reads the row as a sequence
        # Input shape: (Batch, Seq_Len=input_dim, Feature_dim=1)
        self.lstm_enc = nn.LSTM(input_size=1, hidden_size=self.hidden_dim, batch_first=True)
        
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim)
        
        # Decoder: Generates sequence from latent vector
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dim)
        self.lstm_dec = nn.LSTM(input_size=1, hidden_size=self.hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_dim, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Reshape input to (Batch, Sequence_Len, 1)
        # We treat columns as time steps
        x_seq = x.unsqueeze(-1) 
        
        # Encoder
        _, (h_n, _) = self.lstm_enc(x_seq)
        h_n = h_n.squeeze(0) # Take last hidden state
        
        mu, logvar = self.fc_mu(h_n), self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        # We repeat Z to be the initial input for the decoder
        h_dec = self.decoder_input(z).unsqueeze(0) # (1, Batch, Hidden)
        c_dec = torch.zeros_like(h_dec)
        
        # We use a dummy input of zeros for the sequence generation (or teacher forcing)
        # For simplicity in VAE generation, we often decode the static Z
        # Here we just re-run LSTM over a dummy sequence to reconstruct
        dummy_input = torch.zeros(x.size(0), self.input_dim, 1).to(x.device)
        
        out_seq, _ = self.lstm_dec(dummy_input, (h_dec, c_dec))
        recon_seq = self.fc_out(out_seq) # (Batch, Seq, 1)
        
        # Reshape back to tabular (Batch, Features)
        recon_x = torch.sigmoid(recon_seq.squeeze(-1))
        
        return recon_x, mu, logvar
