import pandas as pd
import numpy as np

def create_dummy_data(num_samples=1000):
    np.random.seed(42)
    
    # 1. Age: Random distribution between 18 and 70
    age = np.random.randint(18, 70, num_samples)
    
    # 2. Income: Correlated with Age (older people tend to earn slightly more)
    # Base income + Age factor + Random noise
    income = 30000 + (age * 1000) + np.random.normal(0, 10000, num_samples)
    
    # 3. Daily App Usage (Minutes): Inverse correlation with Age
    # (Younger people spend more time on apps)
    app_usage = 180 - (age * 1.5) + np.random.normal(0, 20, num_samples)
    app_usage = np.maximum(app_usage, 10) # Ensure no negative minutes
    
    # 4. Ad Click Probability (0 to 1): Correlated with Usage
    click_prob = (app_usage / 200) + np.random.normal(0, 0.05, num_samples)
    click_prob = np.clip(click_prob, 0.01, 0.99)
    
    # 5. Total Ad Spend ($): Strongly correlated with Income and Click Probability
    ad_spend = (income / 1000) * click_prob * 5 + np.random.normal(0, 10, num_samples)
    ad_spend = np.maximum(ad_spend, 0)

    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Annual_Income': income.astype(int),
        'Daily_App_Usage_Min': app_usage.astype(int),
        'Ad_Click_Prob': click_prob.round(3),
        'Total_Ad_Spend': ad_spend.round(2)
    })
    
    # Save to CSV
    df.to_csv('dummy_audience.csv', index=False)
    print(f"✅ Generated 'dummy_audience.csv' with {num_samples} rows.")
    print("Columns: Age, Annual_Income, Daily_App_Usage_Min, Ad_Click_Prob, Total_Ad_Spend")

if __name__ == "__main__":
    create_dummy_data()
