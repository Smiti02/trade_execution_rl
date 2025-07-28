# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from environment.trading_env import TradingEnvironment
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
# from environment.trading_env import TradingEnvironment
# from utils.data_loader import load_and_preprocess_data
# from config import config

# def train_ppo():
#     # Load and preprocess data
#     data = load_and_preprocess_data(config.DATA_PATH)
#     train_data, test_data = data[:int(len(data)*config.TRAIN_TEST_SPLIT)], data[int(len(data)*config.TRAIN_TEST_SPLIT):]
    
#     # Create environments
#     train_env = DummyVecEnv([lambda: Monitor(TradingEnvironment(train_data, config))])
#     eval_env = DummyVecEnv([lambda: Monitor(TradingEnvironment(test_data, config))])
    
#     # Create callbacks
#     checkpoint_callback = CheckpointCallback(
#         save_freq=10000,
#         save_path=config.SAVE_PATH,
#         name_prefix="ppo_trade"
#     )
    
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=config.SAVE_PATH,
#         log_path=config.SAVE_PATH,
#         eval_freq=5000,
#         deterministic=True,
#         render=False,
#         n_eval_episodes=config.EVAL_EPISODES
#     )
    
#     # Create model
#     model = PPO(
#         config.POLICY,
#         train_env,
#         verbose=1,
#         learning_rate=config.LEARNING_RATE,
#         gamma=config.GAMMA,
#         batch_size=config.BATCH_SIZE,
#         tensorboard_log="./tensorboard_logs/"
#     )
    
#     # Train model
#     model.learn(
#         total_timesteps=config.TOTAL_TIMESTEPS,
#         callback=[checkpoint_callback, eval_callback]
#     )
    
#     # Save final model
#     model.save(os.path.join(config.SAVE_PATH, "ppo_trade_final"))
    
#     return model

# if __name__ == "__main__":
#     train_ppo()

import sys
import os
import torch
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from environment.trading_env import TradingEnvironment
from utils.data_loader import load_and_preprocess_data
from config import config
#from torch.optim.lr_scheduler import LinearLR

def make_env(data, is_eval=False):
    """Environment creation helper with all required wrappers"""
    env = TradingEnvironment(data, config)
    env = Monitor(env)  # For logging episode stats
    return env
    
    # if not is_eval:
    #     # Only check training env to avoid validation errors
    #     pass
    #     #check_env(env)  # Validate the environment conforms to Gym API
    
    # return env

def train_ppo():
    # Ensure save directory exists
    Path(config.SAVE_PATH).mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    data = load_and_preprocess_data(config.DATA_PATH)

    # Ensure we have enough data for at least one episode
    if len(data) < config.EPISODE_LENGTH:
        raise ValueError(f"Not enough data. Need at least {config.EPISODE_LENGTH} data points, got {len(data)}")

    train_data = data[:int(len(data)*config.TRAIN_TEST_SPLIT)]
    test_data = data[int(len(data)*config.TRAIN_TEST_SPLIT):]
    
    # Create vectorized environments
    train_env = DummyVecEnv([lambda: make_env(train_data)])
    eval_env = DummyVecEnv([lambda: make_env(test_data, is_eval=True)])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, config.EPISODE_LENGTH),  # Ensure at least 1 save per episode
        save_path=config.SAVE_PATH,
        name_prefix="ppo_trade"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.SAVE_PATH,
        log_path=config.SAVE_PATH,
        eval_freq=max(5000, config.EPISODE_LENGTH),  # Ensure at least 1 eval per episode
        deterministic=True,
        render=False,
        n_eval_episodes=config.EVAL_EPISODES
    )
    
    # Create model with MultiInputPolicy for dict observation space
    model = PPO(
        "MultiInputPolicy",  # Changed from config.POLICY to handle dict observations
        #env,
        train_env,
        #ent_coef=0.01,
        learning_rate=config.LEARNING_RATE,
        #learning_rate=LinearLR(0.0003, end_factor=0.1),
        gamma=config.GAMMA,
        batch_size=config.BATCH_SIZE,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),  # Custom network architecture
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True
        },
        n_steps=2048,
        ent_coef=0.01,
        device="auto"  # Automatically uses GPU if available
    )
    
    # Train model with progress bar
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(os.path.join(config.SAVE_PATH, "ppo_trade_final"))
    print(f"Training completed. Model saved to {config.SAVE_PATH}")
    #return model

if __name__ == "__main__":
    train_ppo()