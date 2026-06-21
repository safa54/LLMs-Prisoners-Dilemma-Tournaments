#!/usr/bin/env python3
"""
Analysis script to recalculate tournament standings for custom subgroups of agents.
This simulates how sub-pools of agents performed if they had only played matches against each other.
"""

import json
import argparse
from pathlib import Path

# Group definitions
LLM_AGENTS = {"GPT54Mini", "GPT55", "Gemini35Flash", "ClaudeSonnet4_6"}
COOP_BOTS = {"AlwaysCoopBot", "TitForTatBot", "ForgivingTFT", "TitForTwoTats"}
NON_COOP_BOTS = {"AlwaysDefectBot", "RandomBot", "ManipulativeTFT"}

SUBGROUPS = {
    "1) Only LLM Agents": LLM_AGENTS,
    "2) LLM Agents + Cooperative and Naive Bots": LLM_AGENTS | COOP_BOTS,
    "3) LLM Agents + Exploitative and Non-Cooperative Bots": LLM_AGENTS | NON_COOP_BOTS
}

def load_match_results(input_path):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    data = json.loads(path.read_text(encoding="utf-8"))
    
    # Extract match results list
    if isinstance(data, dict) and "match_results" in data:
        return data["match_results"]
    elif isinstance(data, dict):
        # Fallback to cache structure (dict of match entries)
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, dict) and "agent1_id" in first_val:
            return list(data.values())
            
    raise ValueError("Unrecognized JSON format. Must be a tournament JSON log or match cache JSON.")

def calculate_standings(match_results, target_agents):
    """
    Recalculates standings by only counting matches where BOTH agent1 and agent2 are in target_agents.
    """
    scores = {agent: 0 for agent in target_agents}
    coop_rounds = {agent: 0 for agent in target_agents}
    total_rounds = {agent: 0 for agent in target_agents}
    
    # Store matrix of pair results
    # pair_results[(a1, a2)] = (a1_score, a2_score, a1_coop_rate, a2_coop_rate)
    pair_results = {}

    for match in match_results:
        a1 = match.get("agent1_id")
        a2 = match.get("agent2_id")
        
        # Only include matches where both agents are in our subgroup
        if a1 in target_agents and a2 in target_agents:
            s1 = match.get("agent1_score", 0)
            s2 = match.get("agent2_score", 0)
            cr1 = match.get("agent1_cooperation_rate", 0.0)
            cr2 = match.get("agent2_cooperation_rate", 0.0)
            rounds = match.get("rounds_played", 20)
            
            # Update cumulative statistics
            scores[a1] += s1
            scores[a2] += s2
            
            # cooperation count = rate * rounds
            coop_rounds[a1] += int(round(cr1 * rounds))
            coop_rounds[a2] += int(round(cr2 * rounds))
            
            total_rounds[a1] += rounds
            total_rounds[a2] += rounds
            
            # Store pair matchup results
            pair_results[(a1, a2)] = {
                "scores": (s1, s2),
                "coop_rates": (cr1, cr2),
                "rounds": rounds
            }

    standings = []
    for agent in target_agents:
        tot_score = scores[agent]
        tot_rds = total_rounds[agent]
        avg_score = tot_score / tot_rds if tot_rds > 0 else 0.0
        coop_rate = coop_rounds[agent] / tot_rds if tot_rds > 0 else 0.0
        
        # Determine type
        agent_type = "llm" if agent in LLM_AGENTS else "bot"
        
        standings.append({
            "agent_id": agent,
            "agent_type": agent_type,
            "total_score": tot_score,
            "avg_score": avg_score,
            "coop_rate": coop_rate,
            "rounds_played": tot_rds
        })
        
    # Sort descending by total score
    standings.sort(key=lambda x: x["total_score"], reverse=True)
    return standings, pair_results

def print_table(standings):
    print(f"{'Rank':<5} | {'Agent ID':<20} | {'Type':<6} | {'Total Score':<12} | {'Avg/Round':<10} | {'Coop Rate':<10}")
    print("-" * 75)
    for rank, entry in enumerate(standings):
        print(
            f"{rank+1:<5} | "
            f"{entry['agent_id']:<20} | "
            f"{entry['agent_type']:<6} | "
            f"{entry['total_score']:<12} | "
            f"{entry['avg_score']:<10.2f} | "
            f"{entry['coop_rate']:<10.1%}"
        )
    print("-" * 75)

def format_markdown_table(standings):
    lines = []
    lines.append("| Rank | Agent ID | Type | Total Score | Avg/Round | Coop Rate |")
    lines.append("|---|---|---|---|---|---|")
    for rank, entry in enumerate(standings):
        lines.append(
            f"| {rank+1} | **{entry['agent_id']}** | `{entry['agent_type']}` | {entry['total_score']} | {entry['avg_score']:.2f} | {entry['coop_rate']:.1%} |"
        )
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Analyze tournament results by subgroup.")
    parser.add_argument("-i", "--input", required=True, help="Path to JSON tournament log or cache file.")
    args = parser.parse_args()
    
    try:
        match_results = load_match_results(args.input)
    except Exception as e:
        print(f"Error loading {args.input}: {e}")
        return
        
    # Extract unique agents
    all_agents = set()
    for m in match_results:
        all_agents.add(m.get("agent1_id"))
        all_agents.add(m.get("agent2_id"))
        
    print(f"\n======================================================================")
    print(f" ANALYSIS FOR: {args.input}")
    print(f"======================================================================\n")
    
    # 1) Overall Standings & Pair Matches
    print("--- 1) OVERALL STANDINGS (ALL AGENTS) ---")
    overall_standings, pair_results = calculate_standings(match_results, all_agents)
    print_table(overall_standings)
    
    print("\n--- PAIR MATCH RESULTS (MATCHUPS MATRIX) ---")
    # Print list of pairwise matchups sorted alphabetically
    sorted_pairs = sorted(pair_results.keys())
    for a1, a2 in sorted_pairs:
        res = pair_results[(a1, a2)]
        s1, s2 = res["scores"]
        cr1, cr2 = res["coop_rates"]
        print(f"  * {a1:<15} ({s1:>3} pts, {cr1:>5.1%}) vs {a2:<15} ({s2:>3} pts, {cr2:>5.1%})")
        
    # 2) Sub-pool analyses
    for subgroup_name, subgroup_agents in SUBGROUPS.items():
        # Only keep agents that are actually present in the data
        present_agents = subgroup_agents & all_agents
        if not present_agents:
            continue
            
        print(f"\n======================================================================")
        print(f" SUBGROUP: {subgroup_name}")
        print(f"======================================================================\n")
        
        sub_standings, _ = calculate_standings(match_results, present_agents)
        print_table(sub_standings)

        # Print Markdown table as well for easier copying
        print("\n*Markdown Table Format:*")
        print(format_markdown_table(sub_standings))
        print()

if __name__ == "__main__":
    main()
