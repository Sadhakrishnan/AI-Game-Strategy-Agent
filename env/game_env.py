import functools
import random
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from .reward_engine import RewardEngine
from agents.strategy_agent import StrategyAgent
from agents.coordination_agent import CoordinationAgent
from explainability.decision_explainer import DecisionExplainer
from visualization.renderer import Renderer

def parallel_env(**kwargs):
    return TacticalEnv(**kwargs)

class TacticalEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "name": "tactical_v0",
        "is_parallelizable": True
    }

    def __init__(self, grid_size=10, num_blue=2, num_red=2, max_cycles=100, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_blue = num_blue
        self.num_red = num_red
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.agents = [f"blue_{i}" for i in range(num_blue)] + [f"red_{i}" for i in range(num_red)]
        self.possible_agents = self.agents[:]

        self.reward_engine = RewardEngine()
        self.coordination_agent = CoordinationAgent()
        self.explainer = DecisionExplainer()
        
        self.strategy_agents = {
            agent: StrategyAgent(agent, "blue" if "blue" in agent else "red")
            for agent in self.possible_agents
        }
        
        self.renderer = None
        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = Renderer(grid_size=grid_size)

        # Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4: Attack, 5: Defend, 6: Capture
        self.action_spaces = {agent: Discrete(7) for agent in self.agents}
        
        # Observation: [agent_x, agent_y, health, resources, dist_to_obj, team]
        # Simplified for now: grid representation
        self.observation_spaces = {
            agent: Box(low=0, high=grid_size, shape=(6,), dtype=np.float32)
            for agent in self.agents
        }

        self.state = {}
        self.steps = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.steps = 0
        
        self.agent_positions = {
            agent: [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            for agent in self.agents
        }
        self.agent_health = {agent: 100 for agent in self.agents}
        self.objective_pos = [self.grid_size // 2, self.grid_size // 2]
        
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.steps += 1
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Get team strategies
        obs = self._get_obs()
        blue_strategy = self.coordination_agent.get_team_strategy("blue", obs)
        red_strategy = self.coordination_agent.get_team_strategy("red", obs)
        team_strategies = {**blue_strategy, **red_strategy}

        for agent, action in actions.items():
            if agent not in self.agents:
                continue
            
            # Update high-level strategy
            strategy_agent = self.strategy_agents[agent]
            high_level_action = strategy_agent.select_high_level_action(obs[agent], team_strategies.get(agent))
            
            # Record explanation
            infos[agent]["explanation"] = self.explainer.explain(agent, obs[agent], action, strategy_agent)
            
            # Update opponent models for other agents
            for other_agent in self.agents:
                if (("blue" in agent and "red" in other_agent) or 
                    ("red" in agent and "blue" in other_agent)):
                    self.strategy_agents[other_agent].opponent_model.update(agent, action)

            # Movement
            if action < 4:
                self._move_agent(agent, action)
            # Attack logic (simplified)
            elif action == 4:
                self._attack(agent, rewards)
            # Capture logic
            elif action == 6:
                self._capture(agent, rewards)

        # Check for deaths
        for agent in self.agents[:]:
            if self.agent_health[agent] <= 0:
                terminations[agent] = True
                self.agents.remove(agent)

        # Truncation
        if self.steps >= self.max_cycles:
            for agent in self.agents:
                truncations[agent] = True

        observations = self._get_obs()
        rewards = self.reward_engine.calculate_rewards(self, actions)

        return observations, rewards, terminations, truncations, infos

    def _move_agent(self, agent, action):
        pos = self.agent_positions[agent]
        if action == 0 and pos[1] < self.grid_size - 1: pos[1] += 1 # Up
        elif action == 1 and pos[1] > 0: pos[1] -= 1 # Down
        elif action == 2 and pos[0] > 0: pos[0] -= 1 # Left
        elif action == 3 and pos[0] < self.grid_size - 1: pos[0] += 1 # Right

    def _attack(self, agent, rewards):
        # Damage nearby enemies
        pos = self.agent_positions[agent]
        team = "blue" if "blue" in agent else "red"
        for other_agent in self.agents:
            if team not in other_agent:
                other_pos = self.agent_positions[other_agent]
                dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                if dist <= 1:
                    self.agent_health[other_agent] -= 20
                    rewards[agent] += 5

    def _capture(self, agent, rewards):
        pos = self.agent_positions[agent]
        dist_to_obj = abs(pos[0] - self.objective_pos[0]) + abs(pos[1] - self.objective_pos[1])
        if dist_to_obj == 0:
            rewards[agent] += 50
            # End game if objective captured?
            # For now, just a big reward

    def _get_obs(self):
        observations = {}
        for agent in self.possible_agents:
            if agent in self.agents:
                pos = self.agent_positions[agent]
                health = self.agent_health[agent]
                dist_to_obj = abs(pos[0] - self.objective_pos[0]) + abs(pos[1] - self.objective_pos[1])
                team = 1 if "blue" in agent else 0
                observations[agent] = np.array([pos[0], pos[1], health, 0, dist_to_obj, team], dtype=np.float32)
            else:
                observations[agent] = np.zeros(6, dtype=np.float32)
        return observations

    def render(self):
        if self.renderer:
            self.renderer.render(self)
        elif self.render_mode == "human":
            print(f"Step: {self.steps}")
            print(f"Agent Positions: {self.agent_positions}")
            print(f"Objective: {self.objective_pos}")

    def close(self):
        if self.renderer:
            self.renderer.close()
