from __future__ import annotations

from typing import Literal
from .config import GameConfig
from .agent import Agent


class PDGame:
    def __init__(self, agent1: Agent, agent2: Agent, config: GameConfig):
        self.agent1 = agent1
        self.agent2 = agent2
        self.config = config
        
        # History tracks lists of rounds from each agent's perspective
        self.history1: list[dict] = []
        self.history2: list[dict] = []
        
        self.agent1_score = 0
        self.agent2_score = 0

    def run(self) -> dict:
        payouts = self.config.payouts
        
        # Pre-game communication phase
        msg1 = ""
        msg2 = ""
        if getattr(self.config, "enable_communication", False):
            msg1 = self.agent1.generate_pregame_message(self.agent2.agent_id, payouts)
            msg2 = self.agent2.generate_pregame_message(self.agent1.agent_id, payouts)
            print(f"[*] Pre-game message from {self.agent1.agent_id} to {self.agent2.agent_id}: \"{msg1}\"")
            print(f"[*] Pre-game message from {self.agent2.agent_id} to {self.agent1.agent_id}: \"{msg2}\"")
            self.agent1.receive_pregame_message(msg2)
            self.agent2.receive_pregame_message(msg1)
            
        for round_idx in range(self.config.rounds):
            # 1. Ask agents for decisions
            choice1, reasoning1 = self.agent1.make_decision(self.history1, self.agent2.agent_id, payouts)
            choice2, reasoning2 = self.agent2.make_decision(self.history2, self.agent1.agent_id, payouts)
            
            # Ensure valid choices
            choice1 = "cooperate" if choice1 not in ["cooperate", "defect"] else choice1
            choice2 = "cooperate" if choice2 not in ["cooperate", "defect"] else choice2
            
            # 2. Compute payoffs
            if choice1 == "cooperate" and choice2 == "cooperate":
                pay1, pay2 = payouts.R, payouts.R
            elif choice1 == "cooperate" and choice2 == "defect":
                pay1, pay2 = payouts.S, payouts.T
            elif choice1 == "defect" and choice2 == "cooperate":
                pay1, pay2 = payouts.T, payouts.S
            else: # both defect
                pay1, pay2 = payouts.P, payouts.P
                
            self.agent1_score += pay1
            self.agent2_score += pay2
            
            # 3. Save round to history
            self.history1.append({
                "round": round_idx + 1,
                "my_choice": choice1,
                "opponent_choice": choice2,
                "my_payoff": pay1,
                "opponent_payoff": pay2,
                "my_reasoning": reasoning1,
                "opponent_reasoning": reasoning2,
            })
            
            self.history2.append({
                "round": round_idx + 1,
                "my_choice": choice2,
                "opponent_choice": choice1,
                "my_payoff": pay2,
                "opponent_payoff": pay1,
                "my_reasoning": reasoning2,
                "opponent_reasoning": reasoning1,
            })

        # Calculate cooperation rates
        c1_count = sum(1 for r in self.history1 if r["my_choice"] == "cooperate")
        c2_count = sum(1 for r in self.history2 if r["my_choice"] == "cooperate")
        
        c1_rate = c1_count / self.config.rounds if self.config.rounds > 0 else 0.0
        c2_rate = c2_count / self.config.rounds if self.config.rounds > 0 else 0.0

        return {
            "agent1_id": self.agent1.agent_id,
            "agent2_id": self.agent2.agent_id,
            "agent1_score": self.agent1_score,
            "agent2_score": self.agent2_score,
            "agent1_cooperation_rate": c1_rate,
            "agent2_cooperation_rate": c2_rate,
            "rounds_played": self.config.rounds,
            "agent1_pregame_message": msg1,
            "agent2_pregame_message": msg2,
            "history": self.history1, # perspective of agent1
        }
