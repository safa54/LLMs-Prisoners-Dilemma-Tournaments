from __future__ import annotations

import json
import threading
import concurrent.futures
from pathlib import Path
from typing import Any
from .config import TournamentConfig, AgentConfig, GameConfig
from .agent import (
    Agent, AlwaysCooperateBot, AlwaysDefectBot, TitForTatBot, LLMAgent,
    ForgivingTitForTatBot, ManipulativeTitForTatBot, RandomBot, TitForTwoTatsBot
)
from .game import PDGame


def build_agent(config: AgentConfig) -> Agent:
    if config.agent_type == "bot":
        bot_type = config.bot_type.lower() if config.bot_type else "always_cooperate"
        if bot_type == "always_cooperate":
            return AlwaysCooperateBot(config.agent_id)
        elif bot_type == "always_defect":
            return AlwaysDefectBot(config.agent_id)
        elif bot_type == "tit_for_tat":
            return TitForTatBot(config.agent_id)
        elif bot_type == "forgiving_tit_for_tat":
            return ForgivingTitForTatBot(config.agent_id)
        elif bot_type == "manipulative_tit_for_tat":
            return ManipulativeTitForTatBot(config.agent_id)
        elif bot_type == "random_bot":
            return RandomBot(config.agent_id)
        elif bot_type == "tit_for_two_tats":
            return TitForTwoTatsBot(config.agent_id)
        else:
            raise ValueError(f"Unknown bot type: {config.bot_type}")
    elif config.agent_type == "llm":
        if not config.model_name:
            raise ValueError("model_name must be specified for LLM agents")
        return LLMAgent(
            agent_id=config.agent_id,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning_effort=config.reasoning_effort,
            system_prompt=config.system_prompt,
            api_key=config.api_key,
        )
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")


class MatchCache:
    def __init__(self, cache_file: str | None):
        self.lock = threading.Lock()
        self.cache_file = Path(cache_file) if cache_file else None
        self.cache: dict[str, dict] = {}
        self.load()

    def load(self):
        with self.lock:
            if self.cache_file and self.cache_file.exists():
                try:
                    self.cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"[*] Warning: Could not load match cache from {self.cache_file}: {e}")
                    self.cache = {}

    def save(self):
        with self.lock:
            if self.cache_file:
                try:
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_file.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"[*] Warning: Could not save match cache to {self.cache_file}: {e}")

    def _get_key_and_swapped(self, agent1_id: str, agent2_id: str, game_config: GameConfig) -> tuple[str, bool]:
        sorted_ids = sorted([agent1_id, agent2_id])
        swapped = (agent1_id != sorted_ids[0])
        p = game_config.payouts
        key = f"{sorted_ids[0]}_vs_{sorted_ids[1]}_r{game_config.rounds}_T{p.T}_R{p.R}_P{p.P}_S{p.S}"
        return key, swapped

    def get(self, agent1_id: str, agent2_id: str, game_config: GameConfig) -> dict | None:
        key, swapped = self._get_key_and_swapped(agent1_id, agent2_id, game_config)
        with self.lock:
            if key not in self.cache:
                return None
            
            import copy
            cached_result = self.cache[key]
        if not swapped:
            return copy.deepcopy(cached_result)
        
        # Swap perspectives
        swapped_result = {}
        swapped_result["agent1_id"] = cached_result["agent2_id"]
        swapped_result["agent2_id"] = cached_result["agent1_id"]
        swapped_result["agent1_score"] = cached_result["agent2_score"]
        swapped_result["agent2_score"] = cached_result["agent1_score"]
        swapped_result["agent1_cooperation_rate"] = cached_result["agent2_cooperation_rate"]
        swapped_result["agent2_cooperation_rate"] = cached_result["agent1_cooperation_rate"]
        swapped_result["rounds_played"] = cached_result["rounds_played"]
        if "agent1_pregame_message" in cached_result:
            swapped_result["agent1_pregame_message"] = cached_result["agent2_pregame_message"]
            swapped_result["agent2_pregame_message"] = cached_result["agent1_pregame_message"]
        
        swapped_history = []
        for round_data in cached_result["history"]:
            swapped_round = {
                "round": round_data["round"],
                "my_choice": round_data["opponent_choice"],
                "opponent_choice": round_data["my_choice"],
                "my_payoff": round_data["opponent_payoff"],
                "opponent_payoff": round_data["my_payoff"],
                "my_reasoning": round_data.get("opponent_reasoning", ""),
                "opponent_reasoning": round_data.get("my_reasoning", ""),
            }
            swapped_history.append(swapped_round)
        swapped_result["history"] = swapped_history
        
        return swapped_result

    def set(self, agent1_id: str, agent2_id: str, game_config: GameConfig, result: dict):
        key, swapped = self._get_key_and_swapped(agent1_id, agent2_id, game_config)
        import copy
        result_copy = copy.deepcopy(result)
        
        with self.lock:
            if swapped:
                # Store in alphabetical sorted order (not swapped)
                normalized_result = {}
                normalized_result["agent1_id"] = result_copy["agent2_id"]
                normalized_result["agent2_id"] = result_copy["agent1_id"]
                normalized_result["agent1_score"] = result_copy["agent2_score"]
                normalized_result["agent2_score"] = result_copy["agent1_score"]
                normalized_result["agent1_cooperation_rate"] = result_copy["agent2_cooperation_rate"]
                normalized_result["agent2_cooperation_rate"] = result_copy["agent1_cooperation_rate"]
                normalized_result["rounds_played"] = result_copy["rounds_played"]
                if "agent1_pregame_message" in result_copy:
                    normalized_result["agent1_pregame_message"] = result_copy["agent2_pregame_message"]
                    normalized_result["agent2_pregame_message"] = result_copy["agent1_pregame_message"]
                
                normalized_history = []
                for round_data in result_copy["history"]:
                    norm_round = {
                        "round": round_data["round"],
                        "my_choice": round_data["opponent_choice"],
                        "opponent_choice": round_data["my_choice"],
                        "my_payoff": round_data["opponent_payoff"],
                        "opponent_payoff": round_data["my_payoff"],
                        "my_reasoning": round_data.get("opponent_reasoning", ""),
                        "opponent_reasoning": round_data.get("my_reasoning", ""),
                    }
                    normalized_history.append(norm_round)
                normalized_result["history"] = normalized_history
                self.cache[key] = normalized_result
            else:
                self.cache[key] = result_copy
            
        self.save()


class PDTournament:
    def __init__(self, config: TournamentConfig):
        self.config = config
        self.agents: list[Agent] = [build_agent(ac) for ac in config.agents]
        self.results: list[dict] = []
        self.scores: dict[str, int] = {agent.agent_id: 0 for agent in self.agents}
        self.cooperation_counts: dict[str, int] = {agent.agent_id: 0 for agent in self.agents}
        self.rounds_played: dict[str, int] = {agent.agent_id: 0 for agent in self.agents}
        self.cache = MatchCache(config.match_cache_file)

    def _play_match(self, agent1_config: AgentConfig, agent2_config: AgentConfig) -> tuple[str, str, dict]:
        agent1_id = agent1_config.agent_id
        agent2_id = agent2_config.agent_id
        
        # Check cache first
        game_result = self.cache.get(agent1_id, agent2_id, self.config.game)
        if game_result is None:
            # Instantiate agents
            fresh_agent1 = build_agent(agent1_config)
            fresh_agent2 = build_agent(agent2_config)

            game = PDGame(fresh_agent1, fresh_agent2, self.config.game)
            game_result = game.run()
            
            # Only cache results if there were no API call failures
            has_api_failure = any(
                "API Call failed with error:" in (r.get("my_reasoning") or "") or
                "API Call failed with error:" in (r.get("opponent_reasoning") or "")
                for r in game_result["history"]
            )
            if not has_api_failure:
                self.cache.set(agent1_id, agent2_id, self.config.game, game_result)
            
            print(f"Match completed: {agent1_id} vs {agent2_id}")
        
        return agent1_id, agent2_id, game_result

    def run(self) -> dict[str, Any]:
        num_agents = len(self.agents)
        if num_agents < 2:
            return {"error": "Need at least 2 agents to run a tournament"}

        # Reset state
        self.results = []
        self.scores = {agent.agent_id: 0 for agent in self.agents}
        self.cooperation_counts = {agent.agent_id: 0 for agent in self.agents}
        self.rounds_played = {agent.agent_id: 0 for agent in self.agents}

        # Gather pairs
        match_pairs = []
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                match_pairs.append((self.config.agents[i], self.config.agents[j]))

        # Run in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_pair = {
                executor.submit(self._play_match, a1_cfg, a2_cfg): (a1_cfg, a2_cfg)
                for a1_cfg, a2_cfg in match_pairs
            }
            
            for future in concurrent.futures.as_completed(future_to_pair):
                try:
                    agent1_id, agent2_id, game_result = future.result()
                    
                    # Update tournament state
                    self.results.append(game_result)
                    self.scores[agent1_id] += game_result["agent1_score"]
                    self.scores[agent2_id] += game_result["agent2_score"]
                    
                    # Update cooperation counts
                    c1_count = sum(1 for r in game_result["history"] if r["my_choice"] == "cooperate")
                    c2_count = sum(1 for r in game_result["history"] if r["opponent_choice"] == "cooperate")
                    self.cooperation_counts[agent1_id] += c1_count
                    self.cooperation_counts[agent2_id] += c2_count
                    
                    self.rounds_played[agent1_id] += self.config.game.rounds
                    self.rounds_played[agent2_id] += self.config.game.rounds
                except Exception as exc:
                    print(f"[*] Match generated an exception: {exc}")

        # Compute standings
        standings = []
        for agent in self.agents:
            agent_id = agent.agent_id
            total_score = self.scores[agent_id]
            total_rounds = self.rounds_played[agent_id]
            avg_score_per_round = total_score / total_rounds if total_rounds > 0 else 0.0
            
            c_count = self.cooperation_counts[agent_id]
            cooperation_rate = c_count / total_rounds if total_rounds > 0 else 0.0
            
            standings.append({
                "agent_id": agent_id,
                "agent_type": next(a.agent_type for a in self.config.agents if a.agent_id == agent_id),
                "total_score": total_score,
                "cooperation_rate": cooperation_rate,
                "avg_score_per_round": avg_score_per_round,
            })
            
        # Sort by total score descending
        standings.sort(key=lambda x: x["total_score"], reverse=True)

        summary = {
            "standings": standings,
            "match_results": self.results,
        }

        # Write logs to file if configured
        if self.config.log_file:
            import datetime
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = Path(self.config.log_file)
            
            # Format filename with timestamp suffix
            json_filename = f"{original_path.stem}_{now_str}{original_path.suffix}"
            json_path = original_path.with_name(json_filename)
            
            txt_filename = f"{original_path.stem}_{now_str}.txt"
            txt_path = original_path.with_name(txt_filename)
            
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            
            txt_content = generate_txt_report(summary, now_str)
            txt_path.write_text(txt_content, encoding="utf-8")
            
            summary["saved_json_path"] = str(json_path)
            summary["saved_txt_path"] = str(txt_path)

        return summary


def generate_txt_report(summary: dict, timestamp_str: str) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"      PRISONER'S DILEMMA ROUND-ROBIN TOURNAMENT REPORT")
    lines.append(f"      Timestamp: {timestamp_str}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("STANDINGS:")
    lines.append("-" * 80)
    lines.append(f"{'Rank':<5} | {'Agent ID':<20} | {'Type':<6} | {'Total Score':<12} | {'Avg/Round':<10} | {'Coop Rate':<10}")
    lines.append("-" * 80)
    for rank, entry in enumerate(summary["standings"]):
        lines.append(
            f"{rank+1:<5} | "
            f"{entry['agent_id']:<20} | "
            f"{entry['agent_type']:<6} | "
            f"{entry['total_score']:<12} | "
            f"{entry['avg_score_per_round']:<10.2f} | "
            f"{entry['cooperation_rate']:<10.1%}"
        )
    lines.append("-" * 80)
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("                        MATCHES SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    for match in summary["match_results"]:
        a1, a2 = match["agent1_id"], match["agent2_id"]
        s1, s2 = match["agent1_score"], match["agent2_score"]
        
        history = match["history"]
        choices1 = "".join("+" if r["my_choice"] == "cooperate" else "-" for r in history)
        choices2 = "".join("+" if r["opponent_choice"] == "cooperate" else "-" for r in history)
        
        label1 = f"{a1}({s1})"
        label2 = f"{a2}({s2})"
        max_label_len = max(len(label1), len(label2))
        
        lines.append(f"match: {a1} vs {a2}")
        lines.append(f"{label1:<{max_label_len}} : {choices1}")
        lines.append(f"{label2:<{max_label_len}} : {choices2}")
        lines.append("-" * 40)
    lines.append("")

    lines.append("=" * 80)
    lines.append("                        DETAILED MATCH RESULTS")
    lines.append("=" * 80)
    lines.append("")
    
    for match in summary["match_results"]:
        a1, a2 = match["agent1_id"], match["agent2_id"]
        s1, s2 = match["agent1_score"], match["agent2_score"]
        lines.append(f"Match: {a1} vs {a2}")
        lines.append(f"  Final Score: {a1} ({s1}) - {a2} ({s2})")
        lines.append(f"  Cooperation rate: {a1} ({match['agent1_cooperation_rate']:.0%}) - {a2} ({match['agent2_cooperation_rate']:.0%})")
        if "agent1_pregame_message" in match:
            lines.append(f"  Pre-game Communication:")
            lines.append(f"    - {a1} sent: \"{match['agent1_pregame_message']}\"")
            lines.append(f"    - {a2} sent: \"{match['agent2_pregame_message']}\"")
        lines.append("  Round-by-round details:")
        
        for r in match["history"]:
            lines.append(f"    Round {r['round']}:")
            lines.append(f"      - {a1} chose '{r['my_choice']}'")
            lines.append(f"        Reasoning: {r['my_reasoning']}")
            lines.append(f"      - {a2} chose '{r['opponent_choice']}'")
            lines.append(f"        Reasoning: {r['opponent_reasoning']}")
            lines.append(f"      - Payouts: {a1} got {r['my_payoff']}, {a2} got {r['opponent_payoff']}")
        lines.append("-" * 60)
        lines.append("")
        
    lines.append("=" * 80)
    lines.append("                        LLM PROMPT TEMPLATE REFERENCE")
    lines.append("=" * 80)
    lines.append("Below is the prompt structure sent to LLM agents during the game:")
    lines.append("")
    lines.append("[System Prompt Template]")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("You are agent '{agent_id}', playing a multi-round decision-making game with another player.")
    lines.append("In each round, you and the other player must independently choose either 'cooperate' or 'defect'.")
    lines.append("There is no communication other than seeing each other's choice history.")
    lines.append("")
    lines.append("--- Payout Matrix ---")
    lines.append("- You choose 'cooperate', Opponent chooses 'cooperate': You get R, Opponent gets R")
    lines.append("- You choose 'defect', Opponent chooses 'defect': You get P, Opponent gets P")
    lines.append("- You choose 'cooperate', Opponent chooses 'defect': You get S, Opponent gets T")
    lines.append("- You choose 'defect', Opponent chooses 'cooperate': You get T, Opponent gets S")
    lines.append("")
    lines.append("--- Goal ---")
    lines.append("Maximize your own cumulative score across all rounds. Think strategically about how your choices impact future rounds.")
    lines.append("[Optional: Custom Profile/Directives text defined in system_prompt]")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("")
    lines.append("[User Prompt Template]")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Current Round: {round_num}")
    lines.append("")
    lines.append("--- Game History So Far ---")
    lines.append("Round 1: You chose '{my_choice}', Opponent chose '{opponent_choice}'. Your payoff: X, Opponent payoff: Y.")
    lines.append("Round 2: ...")
    lines.append("")
    lines.append("Submit your choice for the current round ('cooperate' or 'defect').")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("")

    return "\n".join(lines)
