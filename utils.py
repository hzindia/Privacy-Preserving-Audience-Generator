import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    if df_numeric.empty:
        return None, None, None
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    return torch.FloatTensor(scaled_data), scaler, df_numeric.columns

def plot_correlation_comparison(data_dict):
    """
    Dynamically creates subplots with a consistent color scale (-1 to 1).
    """
    num_plots = len(data_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), constrained_layout=True)
    
    if num_plots == 1:
        axes = [axes]

    for i, (name, df) in enumerate(data_dict.items()):
        # vmin=-1, vmax=1 ensures that the color 'Red' always means +1.0 correlation
        # and 'Blue' always means -1.0, regardless of the data range.
        sns.heatmap(
            df.corr(), 
            annot=False, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            ax=axes[i], 
            cbar=True,  # Enables the legend/colorbar
            cbar_kws={"shrink": 0.8} # Makes the bar slightly smaller for better fit
        )
        axes[i].set_title(f"{name} Data", fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
    
    return fig
