class CoordinationAgent:
    def __init__(self):
        self.team_goals = {
            "blue": "capture_objective",
            "red": "defend_objective"
        }
        self.shared_knowledge = {}

    def update_knowledge(self, team, key, value):
        if team not in self.shared_knowledge:
            self.shared_knowledge[team] = {}
        self.shared_knowledge[team][key] = value

    def get_team_strategy(self, team, agents_state):
        # Basic logic: if one agent is close to objective, others should defend or distract
        team_agents = [a for a in agents_state if team in a]
        if not team_agents:
            return "regroup"
            
        # Example: Find the agent closest to the objective
        # Observation index 4 is dist_to_obj
        closest_agent = min(team_agents, key=lambda a: agents_state[a][4])
        
        strategies = {}
        for agent in team_agents:
            if agent == closest_agent:
                strategies[agent] = "capture"
            else:
                strategies[agent] = "support"
        
        return strategies
