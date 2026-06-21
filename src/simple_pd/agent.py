from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Literal
from .config import PayoutMatrix


class Agent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def generate_pregame_message(self, opponent_id: str, payout_matrix: PayoutMatrix) -> str:
        """Generate a pre-game communication message (max 100 characters)."""
        return ""

    def receive_pregame_message(self, message: str):
        """Receive a pre-game communication message from the opponent."""
        pass

    @abstractmethod
    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        """Make a decision for the current round.

        Args:
            game_history: List of past rounds, each round is a dict with keys:
                          "my_choice", "opponent_choice", "my_payoff", "opponent_payoff"
            opponent_id: Name of the opponent
            payout_matrix: Payout values (T, R, P, S)

        Returns:
            Tuple of:
              - Choice: either "cooperate" or "defect"
              - Explanation/Reasoning: text explaining the choice
        """
        raise NotImplementedError


class AlwaysCooperateBot(Agent):
    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        return "cooperate", "Deterministic policy: always cooperate."


class AlwaysDefectBot(Agent):
    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        return "defect", "Deterministic policy: always defect."


class TitForTatBot(Agent):
    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        if not game_history:
            return "cooperate", "Tit For Tat starts by cooperating."
        
        last_opponent_move = game_history[-1]["opponent_choice"]
        return last_opponent_move, f"Tit For Tat copies opponent's last move: {last_opponent_move}."


class LLMAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        reasoning_effort: str | None = None,
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(agent_id)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.received_pregame_message = None

    def receive_pregame_message(self, message: str):
        self.received_pregame_message = message

    def generate_pregame_message(self, opponent_id: str, payout_matrix: PayoutMatrix) -> str:
        try:
            import litellm
            litellm.suppress_helper_message = True
        except ImportError:
            return "Let's cooperate for mutual benefit."

        system_content = (
            f"You are agent '{self.agent_id}', about to play a multi-round Prisoner's Dilemma game with opponent '{opponent_id}'.\n"
            f"Payout Matrix:\n"
            f"- Both cooperate: You get {payout_matrix.R}, Opponent gets {payout_matrix.R}\n"
            f"- Both defect: You get {payout_matrix.P}, Opponent gets {payout_matrix.P}\n"
            f"- You cooperate, Opponent defects: You get {payout_matrix.S}, Opponent gets {payout_matrix.T}\n"
            f"- You defect, Opponent cooperates: You get {payout_matrix.T}, Opponent gets {payout_matrix.S}\n\n"
            f"Before the match starts, you can send ONE pre-game message to your opponent (e.g. to propose cooperation, coordinate, or threaten).\n"
            f"Your message must be extremely concise and MUST NOT exceed 100 characters.\n"
        )
        user_content = (
            "Write the message you want to send to your opponent now. "
            "Output ONLY the raw message string, with no quotes, no extra formatting, and no explanation. Keep it under 100 characters."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "timeout": 60.0,
            "num_retries": 0,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        is_o_series = any(o in self.model_name for o in ["o1", "o3", "o4"])
        if is_o_series:
            kwargs["max_completion_tokens"] = 150
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = 150
            
        try:
            response = litellm.completion(**kwargs)
            msg = response.choices[0].message.content or ""
            msg = msg.strip().replace('"', '').replace("'", "")
            # Enforce max 100 characters limit
            if len(msg) > 100:
                msg = msg[:97] + "..."
            return msg
        except Exception as e:
            print(f"[*] Warning: Pre-game message generation failed for {self.agent_id}: {e}")
            return "Let's cooperate for mutual benefit."

    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        # Import litellm dynamically so it's not a hard import error during package loads
        try:
            import litellm
            litellm.suppress_helper_message = True
        except ImportError:
            raise ImportError(
                "The 'litellm' package is required to run LLM agents. "
                "Please run: pip install litellm"
            )

        # Build prompt history
        history_lines = []
        for idx, round_data in enumerate(game_history):
            history_lines.append(
                f"Round {idx+1}: You chose '{round_data['my_choice']}', Opponent chose '{round_data['opponent_choice']}'. "
                f"Your payoff: {round_data['my_payoff']}, Opponent payoff: {round_data['opponent_payoff']}."
            )
        history_str = "\n".join(history_lines) if history_lines else "No history yet. This is the first round."

        # Setup messages
        system_content = (
            f"You are agent '{self.agent_id}', playing a multi-round decision-making game with another player.\n"
            f"In each round, you and the other player must independently choose either 'cooperate' or 'defect'. "
            f"There is no communication other than seeing each other's choice history.\n\n"
            #f"Be aware that a group of 11 players are present in the game.\n\n"
            f"--- Payout Matrix ---\n"
            f"- You choose 'cooperate', Opponent chooses 'cooperate': You get {payout_matrix.R}, Opponent gets {payout_matrix.R}\n"
            f"- You choose 'defect', Opponent chooses 'defect': You get {payout_matrix.P}, Opponent gets {payout_matrix.P}\n"
            f"- You choose 'cooperate', Opponent chooses 'defect': You get {payout_matrix.S}, Opponent gets {payout_matrix.T}\n"
            f"- You choose 'defect', Opponent chooses 'cooperate': You get {payout_matrix.T}, Opponent gets {payout_matrix.S}\n\n"
            f"--- Goal ---\n"
            f"Your only goal is to maximize your own cumulative score across all rounds. Do not optimize for fairness, morality, or the other player's score unless doing so helps increase your own cumulative score.\n"
            #f"This game is part of an 11-player tournament. Players with low cumulative scores may be eliminated and unable to continue to later games. "
            #f"The players who remain will determine your possible future opponents. If many cooperative players are eliminated, you may face more defecting opponents later. "
#f"Future games also count toward your cumulative score, so choose actions that maximize your long-term total score.\n\n"
        )
        if self.system_prompt:
            system_content += f"\nCustom Profile/Directives:\n{self.system_prompt}\n"

        if getattr(self, "received_pregame_message", None):
            system_content += (
                f"\n--- Pre-Game Communication ---\n"
                f"Before the match started, your opponent '{opponent_id}' sent you this message:\n"
                f"\"{self.received_pregame_message}\"\n\n"
            )

        user_content = (
            f"Current Round: {len(game_history) + 1}\n\n"
            f"--- Game History So Far ---\n"
            f"{history_str}\n\n"
            f"Submit your choice for the current round ('cooperate' or 'defect'). "
            f"You must fill in the 'reasoning' parameter with a detailed explanation of your strategic thinking for this round."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # Define schema for function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_choice",
                    "description": "Submit your choice (cooperate or defect) for this round.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "type": "string",
                                "enum": ["cooperate", "defect"],
                                "description": "Whether you want to cooperate or defect."
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Your detailed strategic thinking behind this choice. You must provide a fresh, context-specific explanation for this round."
                            }
                        },
                        "required": ["choice", "reasoning"]
                    }
                }
            }
        ]

        # LiteLLM completion arguments
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "timeout": 60.0,
            "num_retries": 0,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Handle o-series specific parameters
        is_o_series = any(o in self.model_name for o in ["o1", "o3", "o4"])
        if is_o_series:
            # o-series models do not support standard temperature and max_tokens
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "submit_choice"}}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = litellm.completion(**kwargs)
                
                # Parse response message
                message = response.choices[0].message
                
                # Check if response is empty (no tool calls and no content)
                has_tools = hasattr(message, "tool_calls") and message.tool_calls
                has_content = message.content and message.content.strip()
                
                if not has_tools and not has_content:
                    if attempt < max_retries - 1:
                        print(f"[*] Warning: Empty response from {self.model_name}. Retrying (attempt {attempt + 2}/{max_retries})...")
                        import time
                        time.sleep(1.0)
                        continue
                
                # 1. Parse Tool Call if present
                if has_tools:
                    tool_call = message.tool_calls[0]
                    args = json.loads(tool_call.function.arguments)
                    choice = args.get("choice", "cooperate").strip().lower()
                    
                    # Retrieve reasoning from tool arguments, falling back to message.content if empty/generic
                    reasoning = args.get("reasoning", "").strip()
                    if not reasoning or reasoning.lower() in ["submitted via tool call.", "none", "n/a", ""]:
                        content_str = (message.content or "").strip()
                        if content_str:
                            reasoning = content_str
                        else:
                            # Context-aware fallback to avoid generic placeholder string
                            if choice == "cooperate":
                                if game_history:
                                    last_opp = game_history[-1]["opponent_choice"]
                                    if last_opp == "cooperate":
                                        reasoning = "Continuing cooperation to maintain the mutually beneficial equilibrium."
                                    else:
                                        reasoning = "Cooperating to encourage the opponent to return to mutual cooperation."
                                else:
                                    reasoning = "Cooperating in the first round to signal willingness for mutual cooperation."
                            else:
                                if game_history:
                                    last_opp = game_history[-1]["opponent_choice"]
                                    if last_opp == "defect":
                                        reasoning = "Defecting in response to the opponent's previous defection."
                                    else:
                                        reasoning = "Defecting strategically to maximize payoff or exploit the opponent."
                                else:
                                    reasoning = "Defecting in the first round to avoid risk of exploitation."
                        
                    if choice in ["cooperate", "defect"]:
                        return choice, reasoning

                # 2. Fallback to raw text response parsing if no tool was called
                content = message.content or ""
                content_lower = content.lower()
                
                # Scan for a JSON block or choice in raw text
                if "defect" in content_lower and "cooperate" not in content_lower:
                    return "defect", f"Parsed from text response: {content}"
                elif "cooperate" in content_lower and "defect" not in content_lower:
                    return "cooperate", f"Parsed from text response: {content}"
                
                # Look for JSON structure in content
                try:
                    # Try finding JSON block
                    import re
                    match = re.search(r"\{.*?\}", content, re.DOTALL)
                    if match:
                        json_data = json.loads(match.group(0))
                        choice = json_data.get("choice", "cooperate").strip().lower()
                        reasoning = json_data.get("reasoning", content)
                        if choice in ["cooperate", "defect"]:
                            return choice, reasoning
                except Exception:
                    pass

                # Final safety default
                return "cooperate", f"Default cooperate. Raw response was: {content}"

            except Exception as e:
                # If we have attempts left, retry on exception
                if attempt < max_retries - 1:
                    print(f"[*] Warning: API call to {self.model_name} failed with error: {e}. Retrying (attempt {attempt + 2}/{max_retries})...")
                    import time
                    time.sleep(1.0)
                    continue
                # If we run out of retries, log the error and default to cooperate
                return "cooperate", f"API Call failed with error: {str(e)}. Defaulted to cooperate."


class ForgivingTitForTatBot(Agent):
    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        if not game_history:
            return "cooperate", "Forgiving Tit For Tat starts by cooperating."
        
        last_opponent_move = game_history[-1]["opponent_choice"]
        if last_opponent_move == "defect":
            # Count the number of opponent defections in history
            defect_count = sum(1 for r in game_history if r["opponent_choice"] == "defect")
            if defect_count % 3 == 0:
                return "cooperate", f"Forgiving Tit For Tat: Forgiving the opponent's {defect_count}th defection."
            return "defect", f"Forgiving Tit For Tat: Retaliating opponent's {defect_count}th defection."
        
        return "cooperate", "Forgiving Tit For Tat: Copying opponent's cooperation."


class ManipulativeTitForTatBot(Agent):
    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        if not game_history:
            return "cooperate", "Manipulative Tit For Tat starts by cooperating."
        
        last_opponent_move = game_history[-1]["opponent_choice"]
        if last_opponent_move == "cooperate":
            # Count opponent cooperations
            coop_count = sum(1 for r in game_history if r["opponent_choice"] == "cooperate")
            if coop_count % 3 == 0:
                return "defect", f"Manipulative Tit For Tat: Manipulating on opponent's {coop_count}th cooperation."
            return "cooperate", f"Manipulative Tit For Tat: Copying opponent's {coop_count}th cooperation."
        
        return "defect", "Manipulative Tit For Tat: Copying opponent's defection."


class RandomBot(Agent):
    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        import random
        choice = random.choice(["cooperate", "defect"])
        return choice, f"Random bot chose {choice}."


class TitForTwoTatsBot(Agent):
    def make_decision(
        self,
        game_history: list[dict],
        opponent_id: str,
        payout_matrix: PayoutMatrix
    ) -> tuple[Literal["cooperate", "defect"], str]:
        # Penalizes only if the opponent defects twice in a row
        if len(game_history) >= 2:
            if game_history[-1]["opponent_choice"] == "defect" and game_history[-2]["opponent_choice"] == "defect":
                return "defect", "Tit For Two Tats: Retaliating since opponent defected twice in a row."
        return "cooperate", "Tit For Two Tats: Cooperating as opponent has not defected twice in a row."
