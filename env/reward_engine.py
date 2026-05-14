class RewardEngine:
    def __init__(self):
        self.rewards = {
            "capture_objective": 100,
            "attack_hit": 10,
            "death_penalty": -50,
            "damage_taken": -5,
            "proximity_to_objective": 0.1,
            "cooperation_bonus": 5
        }

    def calculate_rewards(self, env, actions):
        step_rewards = {agent: 0 for agent in env.agents}
        
        for agent in env.agents:
            # 1. Proximity Reward
            pos = env.agent_positions[agent]
            obj_pos = env.objective_pos
            dist = abs(pos[0] - obj_pos[0]) + abs(pos[1] - obj_pos[1])
            # Give a small reward for being closer to the objective
            step_rewards[agent] += (env.grid_size * 2 - dist) * self.rewards["proximity_to_objective"]

            # 2. Capture Reward
            if dist == 0 and actions.get(agent) == 6:
                step_rewards[agent] += self.rewards["capture_objective"]
                # Bonus for team if one agent captures
                team = "blue" if "blue" in agent else "red"
                for teammate in env.agents:
                    if team in teammate and teammate != agent:
                        step_rewards[teammate] += self.rewards["cooperation_bonus"]

            # 3. Combat Rewards (Handled in game_env.py for now, but can be moved here)
            # This engine can be expanded to include complex heuristics
            
        return step_rewards
