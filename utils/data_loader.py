# import numpy as np
# import pandas as pd
# from typing import Dict, List

# def load_and_preprocess_data(file_path: str) -> List[Dict]:
#     """
#     Load and preprocess trading data
#     """
#     # Load raw data
#     df = pd.read_csv(file_path)
    
#     # Convert to list of dictionaries for easier access in the environment
#     data = []
#     for _, row in df.iterrows():
#         data.append({
#             'price': float(row['price']),
#             'volume': float(row['volume']),
#             'bid': float(row['bid']),
#             'ask': float(row['ask']),
#             'spread': float(row['ask'] - row['bid'])
#         })
    
#     return data


import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

def generate_market_data(num_points: int = 1000) -> pd.DataFrame:
    """Generate synthetic market data with realistic patterns"""
    np.random.seed(42)
    
    # Base parameters
    base_price = 100.0
    volatility = 0.001  # 0.1% per minute volatility
    
    # Generate price series with random walk and mean reversion
    prices = base_price + np.cumsum(np.random.normal(0, volatility, num_points))
    prices = np.round(prices, 2)
    
    # Generate volumes with intraday pattern (higher at open/close)
    intraday_pattern = np.sin(np.linspace(-np.pi/2, 3*np.pi/2, num_points)) * 0.5 + 1
    volumes = (intraday_pattern * np.random.uniform(8000, 15000, num_points)).astype(int)
    
    # Generate spreads that widen with volatility
    spreads = np.random.uniform(0.01, 0.05, num_points) * (1 + np.abs(np.random.normal(0, 1, num_points)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': prices,
        'volume': volumes,
        'bid': np.round(prices - spreads/2, 2),
        'ask': np.round(prices + spreads/2, 2),
        'spread': np.round(spreads, 4)
    })
    
    return df

def load_and_preprocess_data(file_path: str) -> List[Dict]:
    """
    Load and preprocess trading data, generating synthetic data if file doesn't exist
    """
    # Generate data if file doesn't exist
    if not Path(file_path).exists():
        print(f"Generating synthetic market data at {file_path}")
        df = generate_market_data(1000)  # Generate 1000 data points
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        if len(df) < 100:
            print(f"Existing data too small ({len(df)} rows), generating synthetic data")
            df = generate_market_data(1000)
            df.to_csv(file_path, index=False)
    
    # Convert to list of dictionaries
    data = []
    for _, row in df.iterrows():
        data.append({
            'price': float(row['price']),
            'volume': float(row['volume']),
            'bid': float(row['bid']),
            'ask': float(row['ask']),
            'spread': float(row['ask'] - row['bid'])
        })
    
    return data