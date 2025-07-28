#import gym
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict
from .market_simulator import MarketImpactSimulator

class TradingEnvironment(gym.Env):
    """Custom OpenAI Gym environment for optimal trade execution"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: np.ndarray, config):
        super(TradingEnvironment, self).__init__()
        
        #self.data = data
        self.data = data * (config.EPISODE_LENGTH // len(data) + 1)  # Repeat data
        self.config = config
        self.current_step = 0
        self.episode_length = config.EPISODE_LENGTH
        self.initial_balance = config.INITIAL_BALANCE
        self.max_order_size = config.MAX_ORDER_SIZE
        
        # Market impact simulator
        self.market_simulator = MarketImpactSimulator(
            config.TEMPORARY_IMPACT, 
            config.PERMANENT_IMPACT
        )
        
        # Action space: discrete actions for order sizes
        # 0: No trade, 1-10: different order sizes (percentage of MAX_ORDER_SIZE)
        self.action_space = spaces.Discrete(11)
        
        # Observation space: market state + portfolio state
        self.observation_space = spaces.Dict({
            "market": spaces.Box(low=0, high=np.inf, shape=(5,)),  # price, volume, etc.
            "portfolio": spaces.Box(low=-np.inf, high=np.inf, shape=(3,))  # position, cash, etc.
        })
        
        # Reset environment
        self.reset()
        
    # def reset(self):
    #     """Reset the environment to initial state"""
    #     self.current_step = 0
    #     self.balance = self.initial_balance
    #     self.position = 0
    #     self.remaining_shares = self.config.MAX_ORDER_SIZE * 10  # Total shares to execute
    #     self.execution_prices = []
    #     self.done = False
        
    #     return self._next_observation()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state with seed support"""
        # We don't use the seed for now, but the parameter must exist
        super().reset(seed=seed)
    
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.remaining_shares = self.config.MAX_ORDER_SIZE * 10  # Total shares to execute
        self.execution_prices = []
        self.done = False
    
        return self._next_observation(), {}
    
    def _next_observation(self) -> Dict:
        """Get the next observation from environment"""
        # Check if we've reached the end of the data
        if self.current_step >= len(self.data):
            self.current_step = 0  # Reset or handle as needed
        
        current_data = self.data[self.current_step]
        
        # Market state
        market_state = np.array([
            current_data['price'], 
            current_data['volume'],
            current_data['bid'],
            current_data['ask'],
            current_data['spread']
        ], dtype=np.float32)
        
        # Portfolio state
        portfolio_state = np.array([
            self.position,
            self.balance,
            self.remaining_shares
        ], dtype=np.float32)
        
        return {
            "market": market_state,
            "portfolio": portfolio_state
        }
    
    def _take_action(self, action: int) -> Tuple[float, float]:
        """Execute the trade action and return execution price and new market price"""
        current_data = self.data[self.current_step]
        current_price = current_data['price']
        market_volume = current_data['volume']
        
        # Determine order size based on action
        if action == 0:  # No trade
            order_size = 0
        else:
            # Scale order size from 10% to 100% of MAX_ORDER_SIZE
            order_size = min(
                (action / 10) * self.max_order_size,
                self.remaining_shares
            )
        
        # Calculate market impact
        execution_price, new_market_price = self.market_simulator.calculate_impact(
            current_price, order_size, market_volume
        )
        
        # Update portfolio
        if order_size > 0:
            cost = order_size * execution_price
            self.balance -= cost
            self.position += order_size
            self.remaining_shares -= order_size
            self.execution_prices.append(execution_price)
        
        return execution_price, new_market_price
    
    def _calculate_reward(self, execution_price: float) -> float:
        """Calculate reward based on execution quality"""
        current_price = self.data[self.current_step]['price']

        # Improved reward components
        price_improvement = (current_price - execution_price) * 10 if execution_price > 0 else 0
        time_penalty = -0.01  # Small penalty per step to encourage faster execution
        inventory_penalty = -abs(self.position) * 0.001

        # Only give slippage penalty if we actually traded
        slippage_penalty = -abs(execution_price - current_price) if execution_price > 0 else 0
        
        # # Slippage from current mid price
        # slippage = execution_price - current_price if execution_price > 0 else 0
        
        # Penalize for slippage and remaining shares
        #reward = -slippage - (self.remaining_shares / (self.config.MAX_ORDER_SIZE * 10))

        reward = price_improvement + time_penalty + inventory_penalty + slippage_penalty
        
        return reward

    def step(self, action):
        """Execute one step in the environment"""
        if self.done:
            raise ValueError("Episode has already completed. Please reset the environment.")

        # Check if we've reached the end of the data
        if self.current_step >= len(self.data):
            observation = self._next_observation()
            return observation, 0, True, True, {"end_of_data": True}
        
        # Execute action
        execution_price, new_market_price = self._take_action(action)
    
        # Update market data with new price (from market impact)
        if execution_price > 0:
            self.data[self.current_step]['price'] = new_market_price
    
        # Calculate reward
        reward = self._calculate_reward(execution_price)
    
        # Update step
        self.current_step += 1
    
        # Check if episode is done
        terminated = False
        truncated = False
    
        if self.current_step >= self.episode_length:
            truncated = True
        if self.remaining_shares <= 0:
            terminated = True
    
        self.done = terminated or truncated
    
        # Prepare info dictionary
        info = {
            "execution_price": execution_price,
            "new_market_price": new_market_price,
            "position": self.position,
            "balance": self.balance,
            "remaining_shares": self.remaining_shares
        }
    
        return self._next_observation(), reward, terminated, truncated, info
    
    # def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
    #     """Execute one step in the environment"""
    #     if self.done:
    #         raise ValueError("Episode has already completed. Please reset the environment.")
            
    #     # Execute action
    #     execution_price, new_market_price = self._take_action(action)
        
    #     # Update market data with new price (from market impact)
    #     if execution_price > 0:
    #         self.data[self.current_step]['price'] = new_market_price
        
    #     # Calculate reward
    #     reward = self._calculate_reward(execution_price)
        
    #     # Update step
    #     self.current_step += 1
        
    #     # Check if episode is done
    #     if self.current_step >= self.episode_length or self.remaining_shares <= 0:
    #         self.done = True
        
    #     # Prepare info dictionary
    #     info = {
    #         "execution_price": execution_price,
    #         "new_market_price": new_market_price,
    #         "position": self.position,
    #         "balance": self.balance,
    #         "remaining_shares": self.remaining_shares
    #     }
        
    #     return self._next_observation(), reward, self.done, info
    
    def render(self, mode='human'):
        """Render the current state of the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Position: {self.position}")
            print(f"Balance: {self.balance}")
            print(f"Remaining Shares: {self.remaining_shares}")
            print(f"Current Price: {self.data[self.current_step]['price']}")