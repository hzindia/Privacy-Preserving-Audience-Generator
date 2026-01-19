import streamlit as st
import pandas as pd
import torch
import torch.optim as optim
from model import TabularVAE, vae_loss_function
from utils import preprocess_data, plot_correlation_comparison

# --- Page Config ---
st.set_page_config(page_title="MirrorData Gen", layout="wide", page_icon="🛡️")

st.title("🛡️ MirrorData: Privacy-Preserving Audience Generator")
st.markdown("""
This tool uses a **Variational Autoencoder (VAE)** to learn the statistical distribution of your data 
and generate a 'Synthetic Audience' that maintains utility while preserving privacy.
""")

# --- Sidebar settings ---
st.sidebar.header("⚙️ Model Configuration")
latent_dim = st.sidebar.slider("Latent Dimension", 2, 20, 10, help="Size of the compressed 'bottleneck' layer.")
epochs = st.sidebar.slider("Training Epochs", 50, 500, 100)
lr = st.sidebar.selectbox("Learning Rate", [1e-2, 1e-3, 1e-4], index=1)
privacy_noise = st.sidebar.slider("Privacy Noise Level", 0.5, 3.0, 1.0, 
                                  help="Higher values = More privacy, but lower data accuracy.")

# --- Main App ---
uploaded_file = st.file_uploader("📂 Upload Real Audience CSV", type="csv")

if uploaded_file is not None:
    # Load Data
    real_df = pd.read_csv(uploaded_file)
    data_tensor, scaler, col_names = preprocess_data(real_df)

    if data_tensor is None:
        st.error("Error: The CSV must contain at least one numerical column.")
    else:
        # Show Data Preview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Real Data Preview")
            st.dataframe(real_df.head(5))
        with col2:
            st.info(f"Detected {data_tensor.shape[1]} numerical columns for training.")

        # Train Button
        if st.button("🚀 Train VAE Model"):
            input_dim = data_tensor.shape[1]
            model = TabularVAE(input_dim, latent_dim)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Training Loop
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_history = []

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward Pass
                recon, mu, logvar = model(data_tensor)
                loss = vae_loss_function(recon, data_tensor, mu, logvar)
                
                # Backward Pass
                loss.backward()
                optimizer.step()
                
                loss_history.append(loss.item())
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Training... Epoch {epoch+1}/{epochs}")

            st.success("Training Complete!")
            
            # Save state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['cols'] = col_names
            st.session_state['real_df'] = real_df # Save for comparison later

            # Plot Loss Curve
            st.line_chart(loss_history)

# --- Generation Section ---
if 'model' in st.session_state:
    st.divider()
    st.header("✨ Generate Synthetic Data")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        num_samples = st.number_input("How many rows to generate?", min_value=10, value=1000, step=100)
        gen_btn = st.button("Generate Samples")
    
    if gen_btn:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        cols = st.session_state['cols']
        
        # Generate
        model.eval()
        with torch.no_grad():
            # Apply privacy noise during sampling
            z = torch.randn(num_samples, latent_dim)
            # Use privacy_noise from slider to scale the random latent vectors
            z = z * privacy_noise 
            synth_scaled = model.decoder(z).numpy()
            
        # Inverse Transform
        synth_data = scaler.inverse_transform(synth_scaled)
        synth_df = pd.DataFrame(synth_data, columns=cols)
        
        # Display Results
        st.subheader("Synthetic Data Preview")
        st.dataframe(synth_df.head())
        
        # Validation Plots
        st.subheader("🔍 Utility Validation: Correlation Matrix")
        st.write("Compare the heatmaps. If they look similar, the model successfully learned the data structure.")
        fig = plot_correlation_comparison(st.session_state['real_df'], synth_df)
        st.pyplot(fig)
        
        # Download
        csv = synth_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Synthetic CSV",
            data=csv,
            file_name="mirror_data_synthetic.csv",
            mime="text/csv",
        )
