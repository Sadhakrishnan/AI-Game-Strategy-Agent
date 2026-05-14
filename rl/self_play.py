import os
import shutil
from stable_baselines3 import PPO
from env.game_env import parallel_env
import supersuit as ss

def run_self_play(iterations=5):
    if not os.path.exists("models/self_play"):
        os.makedirs("models/self_play")
    
    # 1. Initial training if no model exists
    initial_model_path = "models/tactical_agent_final.zip"
    if not os.path.exists(initial_model_path):
        print("No initial model found. Run train.py first.")
        return

    for i in range(iterations):
        print(f"--- Self-Play Iteration {i+1} ---")
        
        # Load the latest model
        model = PPO.load(initial_model_path)
        
        # In a real self-play setup, you'd have one team using a frozen older model
        # and the other team learning. For this simple implementation, we'll
        # just continue training the same policy against itself in the environment.
        
        env = parallel_env(render_mode=None)
        env = ss.black_death_v3(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
        
        model.set_env(env)
        model.learn(total_timesteps=20000)
        
        # Save iteration model
        iter_path = f"models/self_play/tactical_v{i+1}"
        model.save(iter_path)
        
        # Update current best
        shutil.copy(f"{iter_path}.zip", initial_model_path)
        print(f"Iteration {i+1} complete. Model saved to {iter_path}")

if __name__ == "__main__":
    run_self_play()
