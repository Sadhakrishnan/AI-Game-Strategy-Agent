import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import supersuit as ss
from env.game_env import parallel_env

def train():
    # 1. Create the environment
    env = parallel_env(render_mode=None)
    
    # 2. Wrap the environment for SB3 compatibility
    # We use parameter sharing: all agents are treated as a single agent type
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    # 3. Initialize the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        tensorboard_log="./tensorboard_logs/"
    )

    # 4. Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="tactical_agent"
    )

    # 5. Train
    print("Starting training...")
    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
        tb_log_name="PPO_run"
    )

    # 6. Save the final model
    model.save("models/tactical_agent_final")
    print("Training complete and model saved.")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    train()
