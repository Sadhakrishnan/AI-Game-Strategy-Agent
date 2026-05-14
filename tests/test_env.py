from env.game_env import parallel_env
import numpy as np

def test_env_integration():
    print("Testing Environment Integration...")
    env = parallel_env(render_mode="human")
    obs, infos = env.reset()
    
    print(f"Initial agents: {env.agents}")
    
    # Simulate a few steps
    for i in range(5):
        print(f"\n--- Step {i+1} ---")
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        
        for agent in actions.keys():
            if agent in infos:
                explanation = infos[agent].get("explanation", "No explanation")
                print(f"{agent}: {explanation}")
                
    env.close()
    print("\nEnvironment integration test passed!")

if __name__ == "__main__":
    test_env_integration()
