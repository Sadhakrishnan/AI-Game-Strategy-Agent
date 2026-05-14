import streamlit as st
import numpy as np
import time
from env.game_env import parallel_env
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Tactical Dashboard", layout="wide")

st.title("🛡️ AI Game Strategy Agent Platform")

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("Run Simulation"):
    st.session_state.run_sim = True
else:
    if 'run_sim' not in st.session_state:
        st.session_state.run_sim = False

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Gameplay")
    game_container = st.empty()
    
    if st.session_state.run_sim:
        env = parallel_env(render_mode="human")
        obs, infos = env.reset()
        
        for i in range(50):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Since we can't easily embed Pygame in Streamlit here, 
            # we'll display the text-based state and explanations
            with game_container.container():
                st.write(f"Step {i+1}")
                for agent, info in infos.items():
                    if "explanation" in info:
                        st.info(info["explanation"])
            
            time.sleep(0.5)
            if all(terminations.values()):
                break
        env.close()

with col2:
    st.subheader("Training Metrics")
    # Mock data for demonstration
    df = pd.DataFrame(
        np.random.randn(20, 2),
        columns=['Team Blue Reward', 'Team Red Reward']
    )
    st.line_chart(df)
    
    st.subheader("Agent Status")
    st.json({
        "blue_0": {"strategy": "Offensive", "health": 85},
        "blue_1": {"strategy": "Support", "health": 92},
        "red_0": {"strategy": "Defensive", "health": 40},
        "red_1": {"strategy": "Aggressive", "health": 70},
    })

st.subheader("Strategy Explanation Engine")
st.write("The explainable AI layer provides real-time insights into why agents take specific actions.")
