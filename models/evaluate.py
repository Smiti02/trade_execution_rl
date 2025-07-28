import sys
import os
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from environment.trading_env import TradingEnvironment
from utils.data_loader import load_and_preprocess_data
from benchmarks.twap import twap_execution
from benchmarks.vwap import vwap_execution
from config import config
from utils.visualization import plot_results

def evaluate_model(model_path, model_type="ppo"):
    """Evaluate a trained model"""
    try:
        # Load data
        data = load_and_preprocess_data(config.DATA_PATH)
        test_data = data[int(len(data)*config.TRAIN_TEST_SPLIT):]

        # Create vectorized environment
        env = DummyVecEnv([lambda: TradingEnvironment(test_data, config)])
    
        # # Create environment
        # env = TradingEnvironment(test_data, config)
    
        # Load model
        if model_type.lower() == "ppo":
            model = PPO.load(model_path)
        # elif model_type.lower() == "dqn":
        #     model = DQN.load(model_path)
        # else:
        #     raise ValueError("Invalid model type. Use 'ppo' or 'dqn'")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
        # Run evaluation
        obs = env.reset()
        dones = [False]
        rewards = []
        rl_executions = []
        executions = []
        market_prices = []
        step_count = 0
        max_steps = len(test_data)  # Prevent infinite loops
    
        # while not dones[0]:
        #     action, _ = model.predict(obs, deterministic=True)
        #     obs, reward, dones, info = env.step(action)
        #     rewards.append(reward[0])

        #     # Get market price from the first env (index 0)
        #     market_prices.append(env.envs[0].data[env.envs[0].current_step]['price'])

        #     #market_prices.append(env.data[env.current_step]['price'])
        #     # if info["execution_price"] > 0:
        #     #     executions.append(info["execution_price"])
        #     # if 'execution_price' in info and info['execution_price'] > 0:
        #     #     executions.append(info['execution_price'])

        #     if 'execution_price' in infos[0] and infos[0]['execution_price'] > 0:
        #         executions.append(infos[0]['execution_price'])

        while not dones[0] and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            
            rewards.append(reward[0])
            step_count += 1
            
            # Safely access environment data
            if hasattr(env, 'envs') and len(env.envs) > 0:
                current_env = env.envs[0]
                if hasattr(current_env, 'data') and hasattr(current_env, 'current_step'):
                    market_prices.append(current_env.data[current_env.current_step]['price'])
            
            # Safely access execution info
            if infos and len(infos) > 0 and isinstance(infos[0], dict):
                if 'execution_price' in infos[0] and infos[0]['execution_price'] > 0:
                    rl_executions.append(infos[0]['execution_price'])
    
    
        # Calculate benchmark performances
        total_shares = config.MAX_ORDER_SIZE * 10  # Match your environment's setup
        twap_executions = twap_execution(test_data[:step_count], total_shares) if step_count > 0 else []
        vwap_executions = vwap_execution(test_data[:step_count], total_shares) if step_count > 0 else []
        prices = [d['price'] for d in test_data[:len(rewards)]]
        twap_prices = twap_execution(test_data[:len(rewards)], config.MAX_ORDER_SIZE * 10)
        vwap_prices = vwap_execution(test_data[:len(rewards)], config.MAX_ORDER_SIZE * 10)
    
        # # Calculate metrics
        # if len(executions) > 0:
        #     rl_avg_price = np.mean(executions)
        #     twap_avg_price = np.mean(twap_prices)
        #     vwap_avg_price = np.mean(vwap_prices)

        #     print(f"\nRL Execution Performance:")
        #     print(f"- Average Execution Price: {rl_avg_price:.2f}")
        #     print(f"- Total Reward: {sum(rewards):.2f}")
        #     print(f"- Number of Trades: {len(executions)}")
    
    
        #     print(f"RL Average Execution Price: {rl_avg_price}")
        #     print(f"TWAP Average Execution Price: {twap_avg_price}")
        #     print(f"VWAP Average Execution Price: {vwap_avg_price}")
        #     print(f"RL Improvement over TWAP: {(twap_avg_price - rl_avg_price)/twap_avg_price*100:.2f}%")
        #     print(f"RL Improvement over VWAP: {(vwap_avg_price - rl_avg_price)/vwap_avg_price*100:.2f}%")

        # Calculate metrics
        if len(rl_executions) > 5:
            rl_avg = np.mean(rl_executions)
            rl_avg_price = np.mean(executions)
            twap_avg = np.mean(twap_executions) if twap_executions else rl_avg
            vwap_avg = np.mean(vwap_executions) if vwap_executions else rl_avg
            market_avg = np.mean(market_prices) if market_prices else rl_avg
            #market_avg = np.mean(market_prices) if market_prices else rl_avg_price
            #improvement = (market_avg - rl_avg_price)/market_avg*100 if market_avg != 0 else 0
        
            print("\nEvaluation Results:")
            print(f"Steps evaluated: {step_count}")
            print(f"Market Average Price: {market_avg:.2f}")
            print(f"RL Average Execution: {rl_avg:.2f}")
            print(f"TWAP Average: {twap_avg:.2f}")
            print(f"VWAP Average: {vwap_avg:.2f}")
            print(f"RL vs TWAP Improvement: {(twap_avg - rl_avg)/twap_avg*100:.2f}%")
            print(f"RL vs VWAP Improvement: {(vwap_avg - rl_avg)/vwap_avg*100:.2f}%")
            #print(f"Price Improvement: {improvement:.2f}%")
            print(f"Total Reward: {sum(rewards):.2f}")
            print(f"Number of Trades: {len(rl_executions)}")

            # Plot results
            plot_results(
                market_prices=market_prices,
                rl_executions=rl_executions,
                twap_executions=twap_executions,
                vwap_executions=vwap_executions
            )

        else:
            print("Not enough trades for meaningful visualization")

        # # Plot results
        # plot_results(market_prices, executions)

        # # Plot results if we have data
        # if market_prices and executions:
        #     plot_results(market_prices, executions)
    
        # # Plot results
        # plot_results(market_prices, prices, executions, twap_prices, vwap_prices)
    
        # return {
        #     "market_prices": market_prices,
        #     "execution_prices": executions,
        #     "rewards": rewards,
        #     "rl_prices": executions,
        #     "twap_prices": twap_prices,
        #     "vwap_prices": vwap_prices,
        #     "market_prices": prices
        # }

        return {
            "market_prices": market_prices,
            "rl_executions": rl_executions,
            "twap_executions": twap_executions,
            "vwap_executions": vwap_executions,
            #"execution_prices": executions,
            "rewards": rewards
        }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None


if __name__ == "__main__":
    evaluate_model("trained_models/ppo_trade_final.zip", "ppo")