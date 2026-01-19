import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    Selects numerical columns and scales them to [0, 1].
    Returns the tensor, the scaler, and the column names.
    """
    # For this MVP, we focus on numerical data to keep it 'Easy Difficulty'
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    
    if df_numeric.empty:
        return None, None, None
        
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    return torch.FloatTensor(scaled_data), scaler, df_numeric.columns

def plot_correlation_comparison(real_df, synthetic_df):
    """
    Generates a side-by-side heatmap comparison.
    Returns the Matplotlib figure object for Streamlit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(real_df.corr(), annot=False, cmap='coolwarm', ax=axes[0])
    axes[0].set_title("Real Data Correlations")
    
    sns.heatmap(synthetic_df.corr(), annot=False, cmap='coolwarm', ax=axes[1])
    axes[1].set_title("Synthetic Data Correlations")
    
    return fig
