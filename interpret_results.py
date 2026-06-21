#!/usr/bin/env python3
"""
Interpretation script for Prisoner's Dilemma tournament results.
Groups match results per LLM and writes a structured report to a text file.
Supports both tournament output JSONs and match_cache.json files.
"""

import json
import argparse
import datetime
from pathlib import Path

# Common LLM name patterns to auto-detect if no config/standings are available
LLM_KEYWORDS = ["gpt", "gemini", "claude", "sonnet", "llm", "agent", "o1", "o3", "deepseek", "llama"]
BOT_NAMES = ["alwayscoopbot", "alwaysdefectbot", "titfortatbot", "forgivingtft", "manipulativetft", "randombot", "titfortwotats"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpret Prisoner's Dilemma tournament JSON logs or cache."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the tournament report JSON (e.g. simple_pd_tournament_xxx.json) or cache JSON (e.g. match_cache.json)."
    )
    parser.add_argument(
        "-c", "--config",
        help="Optional path to tournament configuration JSON (used to identify LLMs if parsing a cache file)."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the interpreted text report. Defaults to '<input_name>_interpreted.txt'."
    )
    return parser.parse_args()

def detect_llms(match_results, config_data, standings_data):
    """
    Detects which agent IDs belong to LLMs.
    """
    llm_agents = set()
    
    # 1. Try config data first if available
    if config_data and "agents" in config_data:
        for agent in config_data["agents"]:
            if agent.get("agent_type") == "llm":
                llm_agents.add(agent.get("agent_id"))
        if llm_agents:
            return llm_agents
            
    # 2. Try standings data if available
    if standings_data:
        for entry in standings_data:
            if entry.get("agent_type") == "llm":
                llm_agents.add(entry.get("agent_id"))
        if llm_agents:
            return llm_agents

    # 3. Fallback: Auto-detect from match results by inspecting names and reasoning
    all_agents = set()
    for match in match_results:
        all_agents.add(match.get("agent1_id"))
        all_agents.add(match.get("agent2_id"))
        
    for agent_id in all_agents:
        if not agent_id:
            continue
        agent_id_lower = agent_id.lower()
        # If matches keyword and is not in known default bots list
        is_llm = any(kw in agent_id_lower for kw in LLM_KEYWORDS)
        is_known_bot = any(bot == agent_id_lower for bot in BOT_NAMES)
        if is_llm and not is_known_bot:
            llm_agents.add(agent_id)
            
    # If fallback found nothing, check if there's any agent with long reasonings
    if not llm_agents:
        for agent_id in all_agents:
            # Check length of reasonings in history
            for match in match_results:
                if match.get("agent1_id") == agent_id:
                    hist = match.get("history", [])
                    if hist and len(hist[0].get("my_reasoning", "")) > 40:
                        llm_agents.add(agent_id)
                        break
                elif match.get("agent2_id") == agent_id:
                    hist = match.get("history", [])
                    if hist and len(hist[0].get("opponent_reasoning", "")) > 40:
                        llm_agents.add(agent_id)
                        break
                        
    return llm_agents

def load_data(input_path, config_path=None):
    """
    Loads JSON and detects structure (Tournament report or Cache).
    Returns (match_results_list, standings_list, config_data).
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    data = json.loads(input_file.read_text(encoding="utf-8"))
    
    match_results = []
    standings = []
    config_data = None
    
    # Check if this is a tournament output (contains match_results)
    if isinstance(data, dict) and "match_results" in data:
        match_results = data["match_results"]
        standings = data.get("standings", [])
    # Or if it is a Match Cache file (dict of match entries)
    elif isinstance(data, dict):
        # Cache entries should have 'agent1_id' and 'agent2_id'
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, dict) and "agent1_id" in first_val:
            match_results = list(data.values())
        else:
            raise ValueError("Input JSON does not match tournament output format or cache format.")
    else:
        raise ValueError("Invalid JSON root type (expected dictionary).")
        
    # Load optional config file
    if config_path:
        cfg_file = Path(config_path)
        if cfg_file.exists():
            config_data = json.loads(cfg_file.read_text(encoding="utf-8"))
            
    return match_results, standings, config_data

def format_reasoning(reasoning):
    if not reasoning:
        return "N/A"
    # Replace newlines with formatted indentations for readability
    reasoning_clean = reasoning.strip()
    indented = reasoning_clean.replace("\n", "\n          ")
    return indented

def main():
    args = parse_args()
    
    try:
        match_results, standings, config_data = load_data(args.input, args.config)
    except Exception as e:
        print(f"Error loading files: {e}")
        return
        
    llm_agents = detect_llms(match_results, config_data, standings)
    if not llm_agents:
        print("[*] Warning: No LLM agents could be identified. Processing all agents as LLMs.")
        # Fallback to all unique agents
        all_agents = set()
        for match in match_results:
            all_agents.add(match.get("agent1_id"))
            all_agents.add(match.get("agent2_id"))
        llm_agents = all_agents
        
    # Sort LLM list for deterministic grouping
    # If standings exist, order LLMs by their rank in the standings
    if standings:
        standings_order = [s["agent_id"] for s in standings]
        sorted_llms = sorted(list(llm_agents), key=lambda x: standings_order.index(x) if x in standings_order else 999)
    else:
        sorted_llms = sorted(list(llm_agents))
        
    # Prepare output formatting
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append(f"          PRISONER'S DILEMMA INTERPRETED REPORT: GROUPED BY LLM AGENT")
    output_lines.append(f"          Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"          Source File: {args.input}")
    output_lines.append("=" * 100)
    output_lines.append("\n")
    
    for llm_id in sorted_llms:
        output_lines.append("=" * 100)
        output_lines.append(f" LLM AGENT: {llm_id}")
        output_lines.append("=" * 100)
        
        # Gather all matches containing this LLM
        llm_matches = []
        for match in match_results:
            a1 = match.get("agent1_id")
            a2 = match.get("agent2_id")
            if a1 == llm_id or a2 == llm_id:
                llm_matches.append(match)
                
        # Sort matches by opponent name alphabetically
        def get_opponent_id(match):
            return match.get("agent2_id") if match.get("agent1_id") == llm_id else match.get("agent1_id")
            
        llm_matches.sort(key=get_opponent_id)
        
        output_lines.append(f"Total Matches Analyzed: {len(llm_matches)}")
        output_lines.append("-" * 100)
        output_lines.append("\n")
        
        for idx, match in enumerate(llm_matches):
            a1 = match.get("agent1_id")
            a2 = match.get("agent2_id")
            s1 = match.get("agent1_score", 0)
            s2 = match.get("agent2_score", 0)
            cr1 = match.get("agent1_cooperation_rate", 0.0)
            cr2 = match.get("agent2_cooperation_rate", 0.0)
            rounds = match.get("rounds_played", 0)
            
            # Identify roles relative to our focus LLM
            is_agent1 = (a1 == llm_id)
            opponent_id = a2 if is_agent1 else a1
            llm_score = s1 if is_agent1 else s2
            opp_score = s2 if is_agent1 else s1
            llm_coop = cr1 if is_agent1 else cr2
            opp_coop = cr2 if is_agent1 else cr1
            
            # Format match header
            output_lines.append(f" Match {idx + 1}: {llm_id} vs {opponent_id}")
            output_lines.append(f" " + "-" * 50)
            output_lines.append(f"  * Results summary:")
            output_lines.append(f"    - {llm_id:<15} score: {llm_score:<5} (Avg: {llm_score/rounds:.2f}/rd) | Cooperation rate: {llm_coop:.1%}")
            output_lines.append(f"    - {opponent_id:<15} score: {opp_score:<5} (Avg: {opp_score/rounds:.2f}/rd) | Cooperation rate: {opp_coop:.1%}")
            if "agent1_pregame_message" in match:
                my_msg = match["agent1_pregame_message"] if is_agent1 else match["agent2_pregame_message"]
                opp_msg = match["agent2_pregame_message"] if is_agent1 else match["agent1_pregame_message"]
                output_lines.append(f"\n  * Pre-game Messages:")
                output_lines.append(f"    - Sent by {llm_id}: \"{my_msg}\"")
                output_lines.append(f"    - Sent by {opponent_id}: \"{opp_msg}\"")
            output_lines.append("\n  * Round-by-Round Log & Reasoning:")
            
            # Process round history
            history = match.get("history", [])
            for r in history:
                round_num = r.get("round", 0)
                
                # Extract perspective
                if is_agent1:
                    my_choice = r.get("my_choice")
                    opp_choice = r.get("opponent_choice")
                    my_payoff = r.get("my_payoff")
                    opp_payoff = r.get("opponent_payoff")
                    my_reason = r.get("my_reasoning")
                    opp_reason = r.get("opponent_reasoning")
                else:
                    my_choice = r.get("opponent_choice")
                    opp_choice = r.get("my_choice")
                    my_payoff = r.get("opponent_payoff")
                    opp_payoff = r.get("my_payoff")
                    my_reason = r.get("opponent_reasoning")
                    opp_reason = r.get("my_reasoning")
                    
                output_lines.append(f"    Round {round_num}:")
                output_lines.append(f"      - {llm_id} CHOSE '{my_choice}' (Payoff: {my_payoff})")
                output_lines.append(f"        Reasoning: {format_reasoning(my_reason)}")
                output_lines.append(f"      - {opponent_id} CHOSE '{opp_choice}' (Payoff: {opp_payoff})")
                if opp_reason:
                    output_lines.append(f"        Reasoning: {format_reasoning(opp_reason)}")
                output_lines.append("")
                
            output_lines.append("-" * 100)
            output_lines.append("\n")
            
        output_lines.append("\n" * 2)
        
    # Write to file
    if args.output:
        out_path = Path(args.output)
    else:
        in_path = Path(args.input)
        out_name = f"{in_path.stem}_grouped.txt"
        out_path = in_path.parent / out_name
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output_lines), encoding="utf-8")
    
    print(f"[+] Successfully grouped results by LLM and saved to: {out_path}")

if __name__ == "__main__":
    main()
