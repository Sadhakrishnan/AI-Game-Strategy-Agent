import numpy as np

class OpponentModel:
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.opponent_history = {} # agent_id -> list of actions

    def update(self, agent_id, action):
        if agent_id not in self.opponent_history:
            self.opponent_history[agent_id] = []
        
        self.opponent_history[agent_id].append(action)
        if len(self.opponent_history[agent_id]) > self.history_size:
            self.opponent_history[agent_id].pop(0)

    def predict_aggression(self, agent_id):
        if agent_id not in self.opponent_history or not self.opponent_history[agent_id]:
            return 0.5 # Neutral
        
        actions = self.opponent_history[agent_id]
        # Action 4 is 'Attack' in game_env.py
        attacks = actions.count(4)
        return attacks / len(actions)

    def predict_next_move(self, agent_id):
        if agent_id not in self.opponent_history or not self.opponent_history[agent_id]:
            return None
        
        # Simple frequency-based prediction for movement (0-3)
        actions = [a for a in self.opponent_history[agent_id] if a < 4]
        if not actions:
            return None
            
        return max(set(actions), key=actions.count)
