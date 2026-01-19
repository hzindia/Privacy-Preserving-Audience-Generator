# 🛡️ MirrorData: Privacy-Preserving Audience Generator

> **Generate statistically identical synthetic data while preserving user privacy.**

MirrorData is a Generative AI tool built with **PyTorch** and **Streamlit**. It uses a **Variational Autoencoder (VAE)** to learn the underlying distribution of sensitive tabular data (such as user metrics or sensor logs) and generates high-quality synthetic "mirror" data.

This project addresses two critical challenges in Ad Tech and Manufacturing:
1.  **Data Scarcity:** Augmenting small datasets for better model training.
2.  **Data Privacy:** Sharing realistic data with third parties without exposing PII (Personally Identifiable Information).

---

## 🌟 Key Features

* **Variational Autoencoder (VAE) Engine:** Implements a custom neural network in PyTorch with an Encoder-Decoder architecture.
* **Privacy-First Generation:** Includes a tunable "Privacy Noise" parameter to control the trade-off between data utility and user anonymity.
* **Interactive Dashboard:** A full-featured UI built with Streamlit for training and generation without writing code.
* **Real-time Validation:** Automatically generates side-by-side correlation heatmaps to prove the synthetic data preserves valid statistical relationships.
* **Lightweight:** Runs on a standard CPU/Laptop (no heavy GPU required).

---

## 🛠️ Technical Architecture

MirrorData uses a VAE to map input data into a continuous **Latent Space**.
1.  **Encoder:** Compresses the input features (e.g., Age, Income) into a mean ($\mu$) and log-variance ($\sigma$).
2.  **Reparameterization Trick:** Samples a latent vector $z = \mu + \epsilon \cdot \sigma$, where $\epsilon$ is random noise.
3.  **Decoder:** Reconstructs the data from $z$.

**Privacy Mechanism:**
By increasing the randomness in the sampling phase (the $\epsilon$ term), we effectively "blur" the specific details of the original users while maintaining the global population trends.

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

2.  **Create a virtual environment (Optional but Recommended):**
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

1.  **Generate dummy data (for testing):**
    ```bash
    python generate_dummy_data.py
    ```
    *This creates a `dummy_audience.csv` file with 1,000 rows of correlated user data.*

2.  **Launch the Dashboard:**
    ```bash
    streamlit run app.py
    ```
    *The app will open in your browser at `http://localhost:8501`.*

---

## 📂 Project Structure

```text
MirrorData/
├── app.py                  # Main Streamlit dashboard application
├── model.py                # PyTorch VAE architecture & Loss function
├── utils.py                # Data preprocessing & Visualization tools
├── generate_dummy_data.py  # Script to create test datasets
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
