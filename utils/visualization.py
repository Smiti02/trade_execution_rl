import matplotlib.pyplot as plt
import numpy as np

def plot_results(market_prices, rl_executions, twap_executions, vwap_executions):
    """
    Plot comparison of execution strategies
    """
    plt.figure(figsize=(12, 6))
    
    # Market prices
    plt.plot(market_prices, label='Market Price', alpha=0.5)
    
    # Execution prices
    if len(rl_executions) > 0:
        rl_x = np.linspace(0, len(market_prices)-1, len(rl_executions))
        plt.scatter(rl_x, rl_executions, label='RL Execution', color='green', s=10)
    
    if len(twap_executions) > 0:
        twap_x = np.linspace(0, len(market_prices)-1, len(twap_executions))
        plt.scatter(twap_x, twap_executions, label='TWAP Execution', color='red', s=10)
    
    if len(vwap_executions) > 0:
        vwap_x = np.linspace(0, len(market_prices)-1, len(vwap_executions))
        plt.scatter(vwap_x, vwap_executions, label='VWAP Execution', color='blue', s=10)
    
    plt.title('Execution Strategy Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()