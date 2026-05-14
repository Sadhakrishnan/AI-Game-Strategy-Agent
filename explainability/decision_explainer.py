class DecisionExplainer:
    def __init__(self):
        self.action_map = {
            0: "Move Up",
            1: "Move Down",
            2: "Move Left",
            3: "Move Right",
            4: "Attack",
            5: "Defend",
            6: "Capture"
        }

    def explain(self, agent_id, state, action, strategy_agent):
        action_name = self.action_map.get(action, "Unknown Action")
        strategy_reason = strategy_agent.get_explanation()
        
        explanation = f"Agent {agent_id} chose to {action_name}. "
        explanation += f"Strategic Context: {strategy_reason} "
        
        # Add state context
        health = state[2]
        dist = state[4]
        if health < 50:
            explanation += f"Note: Health is low ({health}). "
        if dist == 0:
            explanation += "Note: Agent is at the objective position."
            
        return explanation
