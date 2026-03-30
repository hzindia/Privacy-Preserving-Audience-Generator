# 🛡️ MirrorData: Privacy-Preserving Audience Generator

> **Generate statistically identical synthetic data while preserving user privacy using VAEs, GANs, and LSTMs.**

MirrorData is a comprehensive Generative AI tool built with **PyTorch** and **Streamlit**. It addresses the critical challenge of sharing sensitive tabular data (like user metrics, financial logs, or sensor data) by learning the underlying statistical distribution and generating high-quality "mirror" data that contains **no real user information**.

This project solves two key problems in Ad Tech and Manufacturing:
1.  **Data Scarcity:** Augmenting small datasets for better model training.
2.  **Data Privacy:** Enabling data sharing with third parties without exposing PII (Personally Identifiable Information).

---

## 🌟 Key Features

* **Multi-Model Architecture:**
    * **VAE (Variational Autoencoder):** Best for smooth, privacy-preserving latent representations.
    * **GAN (Generative Adversarial Network):** Uses a Generator-Discriminator game to create highly realistic data distributions.
    * **LSTM (Long Short-Term Memory):** Treats tabular rows as sequences to capture complex inter-feature dependencies (e.g., `Age` -> `Income` -> `Spend`).
* **⚔️ Comparison Workbench:** Train all three models simultaneously and benchmark them side-by-side using correlation heatmaps.
* **Privacy-First Design:** Includes tunable "Privacy Noise" parameters to trade off utility for anonymity.
* **Interactive Dashboard:** No coding required—upload your CSV, select an algorithm, train, and generate synthetic rows instantly.
* **Lightweight:** Optimized to run on standard CPUs/Laptops without heavy GPU requirements.

---

## 🛠️ Technical Implementation

MirrorData implements three distinct generative approaches:

1.  **Tabular VAE:** Compresses input features into a Gaussian Latent Space ($\mu, \sigma$) and samples new data from that distribution.
2.  **Tabular GAN:** A **Generator** tries to create fake data to fool a **Discriminator**, while the Discriminator learns to distinguish real from fake.
3.  **Seq-2-Seq LSTM:** Treats a row of data as a time series sequence, capturing the sequential relationship between features.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/MirrorData.git](https://github.com/yourusername/MirrorData.git)
    cd MirrorData
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Linux/Mac
    source venv/bin/activate
    # Windows
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Generate dummy data (Optional):**
    ```bash
    python generate_dummy_data.py
    ```
    *Creates a `dummy_audience.csv` file with correlated user data for testing.*

2.  **Launch the Dashboard:**
    ```bash
    streamlit run app.py
    ```
    *The app will open in your browser at `http://localhost:8501`.*

---

## 🕹️ Usage Guide

1.  **Upload Data:** Drag and drop any CSV file containing numerical data.
2.  **Select Algorithm:**
    * Choose **VAE**, **GAN**, or **LSTM** to focus on one specific architecture.
    * Choose **Compare All** to train all three and benchmark them.
3.  **Configure:** Adjust `Latent Dimension` and `Epochs` via the sidebar.
4.  **Train:** Click **🚀 Start Training**. Watch the loss curves update in real-time.
5.  **Generate & Compare:**
    * Enter the number of synthetic rows you need.
    * Click **Generate**.
    * View the **Correlation Matrix Comparison** to see which model best captured the relationships in your real data.
6.  **Download:** Export the synthetic datasets for use in your projects.

---

## 📂 Project Structure

```text
MirrorData/
├── app.py                  # Main Streamlit Dashboard UI
├── model.py                # PyTorch Architectures (VAE, GAN, LSTM)
├── utils.py                # Visualization, Preprocessing & Plotting logic
├── generate_dummy_data.py  # Script to generate test CSVs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
