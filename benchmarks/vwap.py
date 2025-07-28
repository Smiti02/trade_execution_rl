import numpy as np

def vwap_execution(data, total_shares):
    """
    Volume-Weighted Average Price execution strategy
    """
    prices = np.array([d['price'] for d in data])
    volumes = np.array([d['volume'] for d in data])
    
    # Calculate cumulative volume
    cum_volume = np.cumsum(volumes)
    total_volume = cum_volume[-1]
    
    # Target cumulative volume at each step
    target_cum_volume = np.linspace(0, total_volume, len(data))
    
    # Find the indices where we would execute trades
    exec_indices = np.searchsorted(cum_volume, target_cum_volume)
    exec_indices = np.clip(exec_indices, 0, len(prices)-1)
    
    # Get execution prices
    execution_prices = prices[exec_indices]
    
    # Adjust for total shares
    trade_sizes = np.diff(target_cum_volume, prepend=0) / total_volume * total_shares
    execution_prices = []
    remaining_shares = total_shares
    
    for price, size in zip(prices, trade_sizes):
        if remaining_shares <= 0:
            break
            
        actual_size = min(size, remaining_shares)
        execution_prices.extend([price] * int(actual_size))
        remaining_shares -= actual_size
    
    return execution_prices