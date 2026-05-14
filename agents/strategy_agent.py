from .opponent_model import OpponentModel
from .coordination_agent import CoordinationAgent

class StrategyAgent:
    def __init__(self, agent_id, team):
        self.agent_id = agent_id
        self.team = team
        self.opponent_model = OpponentModel()
        self.current_strategy = "exploring"

    def select_high_level_action(self, state, team_strategy):
        health = state[2]
        dist_to_obj = state[4]
        
        # 1. Survival check
        if health < 30:
            self.current_strategy = "retreat"
            return "retreat"
            
        # 2. Team coordination check
        if team_strategy == "capture":
            self.current_strategy = "offensive"
        elif team_strategy == "support":
            self.current_strategy = "defensive"
            
        # 3. Objective proximity
        if dist_to_obj < 2:
            self.current_strategy = "capture"
            
        return self.current_strategy

    def get_explanation(self):
        explanations = {
            "retreat": "Health is low, prioritizing survival.",
            "offensive": "Team strategy is set to offensive to capture the objective.",
            "defensive": "Supporting teammates by holding a defensive position.",
            "capture": "Close to the objective, attempting to capture."
        }
        return explanations.get(self.current_strategy, "Following default exploration policy.")
