# AI Game Strategy Agent

An adaptive multi-agent strategic intelligence platform that uses reinforcement learning, self-play training, opponent modeling, and explainable AI to simulate autonomous gameplay agents capable of planning, coordination, and dynamic strategy adaptation.

---

## 🚀 Features

- Multi-agent reinforcement learning
- Autonomous gameplay agents
- Strategic planning and coordination
- Opponent behavior modeling
- Self-play training
- Explainable AI decisions
- Real-time gameplay visualization
- Reward optimization and analytics
- Adaptive enemy strategies
- Interactive training dashboard

---

## 🎯 Project Goal

The system can:

✅ Train autonomous game agents using reinforcement learning  
✅ Support cooperative and competitive gameplay  
✅ Coordinate multi-agent strategies  
✅ Adapt to opponent behavior dynamically  
✅ Learn through self-play training  
✅ Explain AI decisions and strategic reasoning  
✅ Visualize gameplay and reward evolution  
✅ Track performance and learning metrics  

### Example Output

```json
{
  "agent_action": "defend",
  "reward": 12,
  "strategy": "protect objective",
  "enemy_prediction": "left flank attack",
  "explanation": "The agent chose defense because health was low and the objective was under attack."
}
```

---

## 🧠 System Architecture

```text
Game Environment
        ↓
State Observation Layer
        ↓
RL Decision Engine
        ↓
Strategy Planning Agent
        ↓
Multi-Agent Coordination System
        ↓
Opponent Modeling Engine
        ↓
Action Selection
        ↓
Reward System
        ↓
Policy Learning & Updates
        ↓
Visualization Dashboard
        ↓
Metrics & Analytics
```

---

## 🛠️ Tech Stack

### Reinforcement Learning
- Stable-Baselines3
- PyTorch
- PPO
- DQN
- A3C
- SAC

### Multi-Agent Systems
- PettingZoo
- Ray RLlib

### Backend
- FastAPI

### Frontend / Visualization
- Streamlit
- Pygame
- TensorBoard
- Weights & Biases

### AI / Memory
- Vector Memory
- Replay Buffers
- Embedding-based Episodic Memory

---

## 🎮 Game Environment

Custom grid-based tactical strategy environment built using:
- OpenAI Gym
- PettingZoo

### Supported Mechanics
- Capture the flag
- Resource collection
- Tactical movement
- Territory control
- Cooperative defense
- Attack and retreat

### Action Space
- move_up
- move_down
- move_left
- move_right
- attack
- defend
- collect_resource
- capture_objective

---

## 🔥 Core Components

### 🧠 Reinforcement Learning Engine
Implements RL training using:
- PPO (Primary)
- DQN
- SAC
- A3C

Training loop:
```text
Observe State
    ↓
Select Action
    ↓
Execute Action
    ↓
Receive Reward
    ↓
Update Policy
```

---

### 🤝 Multi-Agent Coordination System
Supports:
- Shared rewards
- Team-based strategies
- Coordinated attacks
- Defensive formations
- Decentralized execution

Example:
> Agent A distracts enemy while Agent B captures the objective.

---

### 🎯 Opponent Modeling Engine
Tracks:
- Enemy movement patterns
- Attack frequency
- Aggressive vs defensive behavior
- Strategic tendencies

Example Insight:
> “Enemy frequently attacks from the left flank.”

---

### 🧩 Strategic Planning Agent
Performs high-level reasoning such as:
- Objective prioritization
- Attack vs defense switching
- Tactical planning
- Strategic action generation

Example:
> “Retreat and defend due to low health and nearby enemy presence.”

---

### 🔁 Self-Play Training
Agents:
- Compete against previous versions
- Learn evolving strategies
- Improve policies autonomously

Pipeline:
```text
Agent v1
    ↓
Self-play Competition
    ↓
Policy Improvement
    ↓
Strategy Evolution
```

---

### 🧠 Episodic Memory System
Stores:
- Successful strategies
- Opponent tendencies
- Important game states
- Previous rewards

Uses:
- Replay buffers
- Embedding memory
- Strategic state tracking

---

### 🔍 Explainable AI Layer
Explains:
- Why an action was selected
- Reward contribution
- Strategic intent

Example:
> “The agent chose defense because health was critically low and enemy proximity was high.”

Optional:
- Attention visualization
- Reward attribution maps

---

## 📊 Visualization Dashboard

Features:
- Live gameplay rendering
- Agent movement tracking
- Reward progression graphs
- Heatmaps
- Strategy overlays
- Opponent behavior visualization
- Win/loss statistics

Uses:
- Streamlit
- Pygame
- TensorBoard

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/game-strategy-agent.git

cd game-strategy-agent

pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Train Agents
```bash
python rl/train.py
```

### Start API
```bash
uvicorn api.main:app --reload
```

### Launch Dashboard
```bash
streamlit run frontend/app.py
```

---

## 🌐 API Endpoints

```http
POST /train
POST /simulate
GET  /metrics
GET  /agents
POST /reset
```

---

## 🔥 Advanced Features

- Hierarchical reinforcement learning
- Curriculum learning
- Procedural enemy generation
- Adaptive difficulty scaling
- Cooperative + competitive gameplay
- Natural language strategic commands
- Generative opponent creation

---

## 📊 Metrics

- Average Reward
- Win Rate
- Convergence Stability
- Cooperation Efficiency
- Adaptation Score
- Opponent Prediction Accuracy
