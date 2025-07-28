import numpy as np

def twap_execution(data, total_shares):
    """
    Time-Weighted Average Price execution strategy
    """
    prices = [d['price'] for d in data]
    times = range(len(prices))
    
    # Equal trades at each time step
    trades = np.linspace(0, total_shares, len(prices))
    trade_sizes = np.diff(trades, prepend=0)
    
    # Calculate execution prices with temporary impact
    execution_prices = []
    remaining_shares = total_shares
    
    for i, (price, size) in enumerate(zip(prices, trade_sizes)):
        if remaining_shares <= 0:
            break
            
        actual_size = min(size, remaining_shares)
        execution_prices.extend([price] * int(actual_size))
        remaining_shares -= actual_size
    
    return execution_prices