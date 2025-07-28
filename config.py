import numpy as np
import os

class Config:
    # Data parameters
    DATA_PATH = "data/historical_trades.csv"
    #DATA_PATH = os.path.join(os.path.dirname(__file__), "data/historical_trades.csv")
    TRAIN_TEST_SPLIT = 0.8
    
    # Environment parameters
    INITIAL_BALANCE = 1000000  # Initial portfolio value
    MAX_ORDER_SIZE = 10000     # Maximum shares per order
    EPISODE_LENGTH = 1000      # Number of steps per episode
    #EPISODE_LENGTH = 5      # Number of steps per episode
    
    # RL parameters
    POLICY = "MlpPolicy"
    TOTAL_TIMESTEPS = 100000
    LEARNING_RATE = 0.0003
    GAMMA = 0.99               # Discount factor
    BATCH_SIZE = 64
    
    # Market impact parameters
    TEMPORARY_IMPACT = 0.001   # Temporary impact coefficient
    PERMANENT_IMPACT = 0.0005  # Permanent impact coefficient
    
    # Evaluation parameters
    #EVAL_EPISODES = 10
    EVAL_EPISODES = 5
    SAVE_PATH = "trained_models/"
    
config = Config()