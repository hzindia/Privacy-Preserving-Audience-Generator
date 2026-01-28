import streamlit as st
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from model import TabularVAE, vae_loss_function, Generator, Discriminator, TabularLSTM_VAE
from utils import preprocess_data, plot_correlation_comparison

# --- Page Config ---
st.set_page_config(page_title="MirrorData: AI Workbench", layout="wide", page_icon="🧪")

st.title("🧪 MirrorData: Generative AI Workbench")
st.markdown("Train, Generate, and **Compare** VAE, GAN, and LSTM algorithms.")

# --- Sidebar settings ---
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.radio(
    "Select Algorithm", 
    ["VAE", "GAN", "LSTM (Seq-to-Seq)", "Compare All"], 
    index=0
)

latent_dim = st.sidebar.slider("Latent Dimension", 2, 20, 10)
epochs = st.sidebar.slider("Training Epochs", 50, 1000, 100)
lr = st.sidebar.selectbox("Learning Rate", [0.0002, 0.001, 0.01], index=1)

# --- Session State Init ---
if 'vae_model' not in st.session_state: st.session_state['vae_model'] = None
if 'gan_generator' not in st.session_state: st.session_state['gan_generator'] = None
if 'lstm_model' not in st.session_state: st.session_state['lstm_model'] = None

# --- Main App ---
uploaded_file = st.file_uploader("📂 Upload Real Data CSV", type="csv")

if uploaded_file is not None:
    real_df = pd.read_csv(uploaded_file)
    data_tensor, scaler, col_names = preprocess_data(real_df)
    input_dim = data_tensor.shape[1]

    # Show Data Preview
    with st.expander("📄 View Real Data Preview", expanded=True):
        st.dataframe(real_df.head(5))

    # ==========================
    # TRAINING SECTION
    # ==========================
    st.divider()
    st.header("1. Training Phase")
    
    if st.button("🚀 Start Training"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- TRAIN VAE ---
        if model_choice in ["VAE", "Compare All"]:
            status_text.text("Training VAE...")
            vae = TabularVAE(input_dim, latent_dim)
            optimizer_vae = optim.Adam(vae.parameters(), lr=lr)
            
            for epoch in range(epochs):
                vae.train()
                optimizer_vae.zero_grad()
                recon, mu, logvar = vae(data_tensor)
                loss = vae_loss_function(recon, data_tensor, mu, logvar)
                loss.backward()
                optimizer_vae.step()
                if model_choice != "Compare All": progress_bar.progress((epoch + 1) / epochs)
            
            st.session_state['vae_model'] = vae
            st.success(f"✅ VAE Trained (Loss: {loss.item():.4f})")

        # --- TRAIN GAN ---
        if model_choice in ["GAN", "Compare All"]:
            status_text.text("Training GAN...")
            generator = Generator(latent_dim, input_dim)
            discriminator = Discriminator(input_dim)
            
            opt_g = optim.Adam(generator.parameters(), lr=lr)
            opt_d = optim.Adam(discriminator.parameters(), lr=lr)
            criterion = nn.BCELoss()
            
            for epoch in range(epochs):
                # 1. Train Discriminator
                opt_d.zero_grad()
                real_labels = torch.ones(data_tensor.size(0), 1)
                fake_labels = torch.zeros(data_tensor.size(0), 1)
                
                d_real = discriminator(data_tensor)
                d_real_loss = criterion(d_real, real_labels)
                
                z = torch.randn(data_tensor.size(0), latent_dim)
                fake_data = generator(z)
                d_fake = discriminator(fake_data.detach())
                d_fake_loss = criterion(d_fake, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                opt_d.step()
                
                # 2. Train Generator
                opt_g.zero_grad()
                d_fake_preds = discriminator(fake_data)
                g_loss = criterion(d_fake_preds, real_labels)
                g_loss.backward()
                opt_g.step()
                
                if model_choice != "Compare All": progress_bar.progress((epoch + 1) / epochs)
                
            st.session_state['gan_generator'] = generator
            st.success(f"✅ GAN Trained (G Loss: {g_loss.item():.4f})")
            
        # --- TRAIN LSTM ---
        if model_choice in ["LSTM (Seq-to-Seq)", "Compare All"]:
            status_text.text("Training LSTM...")
            lstm = TabularLSTM_VAE(input_dim, latent_dim)
            opt_lstm = optim.Adam(lstm.parameters(), lr=lr)
            
            for epoch in range(epochs):
                lstm.train()
                opt_lstm.zero_grad()
                # LSTM forward handles reshaping internally
                recon, mu, logvar = lstm(data_tensor) 
                loss = vae_loss_function(recon, data_tensor, mu, logvar)
                loss.backward()
                opt_lstm.step()
                if model_choice != "Compare All": progress_bar.progress((epoch + 1) / epochs)
                
            st.session_state['lstm_model'] = lstm
            st.success(f"✅ LSTM Trained (Loss: {loss.item():.4f})")

        # Save shared tools
        st.session_state['scaler'] = scaler
        st.session_state['cols'] = col_names
        st.session_state['real_df'] = real_df

    # ==========================
    # GENERATION & COMPARISON
    # ==========================
    if any(st.session_state[k] for k in ['vae_model', 'gan_generator', 'lstm_model']):
        st.divider()
        st.header("2. Generation & Comparison")
        
        num_samples = st.number_input("Samples to Generate", value=1000, step=100)
        
        if st.button("Generate Synthetic Data"):
            comparison_dict = {'Real': st.session_state['real_df']}
            scaler = st.session_state['scaler']
            cols = st.session_state['cols']
            
            # Generate VAE Data
            if st.session_state['vae_model']:
                vae = st.session_state['vae_model']
                with torch.no_grad():
                    z = torch.randn(num_samples, latent_dim)
                    synth_scaled = vae.decoder(z).numpy()
                comparison_dict['VAE'] = pd.DataFrame(scaler.inverse_transform(synth_scaled), columns=cols)
                
            # Generate GAN Data
            if st.session_state['gan_generator']:
                gen = st.session_state['gan_generator']
                with torch.no_grad():
                    z = torch.randn(num_samples, latent_dim)
                    synth_scaled = gen(z).numpy()
                comparison_dict['GAN'] = pd.DataFrame(scaler.inverse_transform(synth_scaled), columns=cols)

            # Generate LSTM Data
            if st.session_state['lstm_model']:
                lstm = st.session_state['lstm_model']
                with torch.no_grad():
                    z = torch.randn(num_samples, latent_dim)
                    # For LSTM-VAE decoding, we replicate the decoder logic manually
                    h_dec = lstm.decoder_input(z).unsqueeze(0)
                    c_dec = torch.zeros_like(h_dec)
                    dummy_input = torch.zeros(num_samples, input_dim, 1)
                    
                    out_seq, _ = lstm.lstm_dec(dummy_input, (h_dec, c_dec))
                    recon_seq = lstm.fc_out(out_seq)
                    synth_scaled = torch.sigmoid(recon_seq.squeeze(-1)).numpy()
                    
                comparison_dict['LSTM'] = pd.DataFrame(scaler.inverse_transform(synth_scaled), columns=cols)

            # --- Visual Comparison ---
            st.subheader("📊 Correlation Matrix Comparison")
            fig = plot_correlation_comparison(comparison_dict)
            st.pyplot(fig)
            
            # --- Downloads ---
            st.subheader("📥 Download Data")
            cols = st.columns(len(comparison_dict)-1)
            idx = 0
            for name, df in comparison_dict.items():
                if name == 'Real': continue
                csv = df.to_csv(index=False).encode('utf-8')
                cols[idx].download_button(f"Download {name} CSV", csv, f"{name.lower()}_data.csv", "text/csv")
                idx += 1
