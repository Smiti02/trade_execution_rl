import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.trading_env import TradingEnvironment
from utils.data_loader import load_and_preprocess_data
from config import config

def train_dqn():
    # Load and preprocess data
    data = load_and_preprocess_data(config.DATA_PATH)
    train_data, test_data = data[:int(len(data)*config.TRAIN_TEST_SPLIT)], data[int(len(data)*config.TRAIN_TEST_SPLIT):]
    
    # Create environments
    train_env = DummyVecEnv([lambda: Monitor(TradingEnvironment(train_data, config))])
    eval_env = DummyVecEnv([lambda: Monitor(TradingEnvironment(test_data, config))])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=config.SAVE_PATH,
        name_prefix="dqn_trade"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.SAVE_PATH,
        log_path=config.SAVE_PATH,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=config.EVAL_EPISODES
    )
    
    # Create model
    model = DQN(
        config.POLICY,
        train_env,
        verbose=1,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        batch_size=config.BATCH_SIZE,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Train model
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    model.save(os.path.join(config.SAVE_PATH, "dqn_trade_final"))
    
    return model

if __name__ == "__main__":
    train_dqn()