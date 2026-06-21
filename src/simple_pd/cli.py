from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from .config import TournamentConfig
from .tournament import PDTournament


def get_default_config() -> dict:
    return {
        "game": {
            "rounds": 5,
            "payouts": {
                "T": 5,
                "R": 3,
                "P": 1,
                "S": 0
            }
        },
        "agents": [
            {
                "agent_id": "AlwaysCoopBot",
                "agent_type": "bot",
                "bot_type": "always_cooperate"
            },
            {
                "agent_id": "AlwaysDefectBot",
                "agent_type": "bot",
                "bot_type": "always_defect"
            },
            {
                "agent_id": "TitForTatBot",
                "agent_type": "bot",
                "bot_type": "tit_for_tat"
            }
        ],
        "log_file": "logs/simple_pd_tournament.json"
    }


def main():
    parser = argparse.ArgumentParser(description="Run a round-robin Prisoner's Dilemma tournament.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simple_pd.json",
        help="Path to the tournament configuration JSON file (default: configs/simple_pd.json)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override the number of rounds in each match"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Override the log file output path"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Override the match cache file path"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    
    # If config file doesn't exist, generate the default one
    if not config_path.exists():
        print(f"[*] Configuration file '{config_path}' not found.")
        print(f"[*] Generating default configuration at '{config_path}'...")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(get_default_config(), indent=2), encoding="utf-8")
        
    try:
        config = TournamentConfig.from_json(config_path)
    except Exception as e:
        print(f"[!] Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply overrides
    if args.rounds is not None:
        config.game.rounds = args.rounds
    if args.log_file is not None:
        config.log_file = args.log_file
    if args.cache is not None:
        config.match_cache_file = args.cache

    print("=" * 60)
    print("      PRISONER'S DILEMMA ROUND-ROBIN TOURNAMENT")
    print("=" * 60)
    print(f"[*] Loaded agents: {', '.join(a.agent_id for a in config.agents)}")
    print(f"[*] Game rounds per match: {config.game.rounds}")
    print(f"[*] Payout matrix: T={config.game.payouts.T}, R={config.game.payouts.R}, P={config.game.payouts.P}, S={config.game.payouts.S}")
    print("-" * 60)
    print("[*] Running tournament matches...")
    
    tournament = PDTournament(config)
    results = tournament.run()
    
    print("\n" + "=" * 60)
    print("                    FINAL STANDINGS")
    print("=" * 60)
    print(f"{'Rank':<5} | {'Agent ID':<20} | {'Type':<6} | {'Total Score':<12} | {'Avg/Round':<10} | {'Coop Rate':<10}")
    print("-" * 60)
    for rank, entry in enumerate(results["standings"]):
        print(
            f"{rank+1:<5} | "
            f"{entry['agent_id']:<20} | "
            f"{entry['agent_type']:<6} | "
            f"{entry['total_score']:<12} | "
            f"{entry['avg_score_per_round']:<10.2f} | "
            f"{entry['cooperation_rate']:<10.1%}"
        )
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("                  MATCH-BY-MATCH RESULTS")
    print("=" * 60)
    for res in results["match_results"]:
        a1, a2 = res["agent1_id"], res["agent2_id"]
        s1, s2 = res["agent1_score"], res["agent2_score"]
        c1, c2 = res["agent1_cooperation_rate"], res["agent2_cooperation_rate"]
        print(f"Match: {a1} vs {a2}")
        print(f"  Score: {a1} ({s1}) - {a2} ({s2})")
        print(f"  Cooperation rate: {a1} ({c1:.0%}) - {a2} ({c2:.0%})")
        print("-" * 40)
        
    if "saved_json_path" in results:
        print(f"[*] Detailed JSON log saved to: {results['saved_json_path']}")
        print(f"[*] Human-readable TXT report saved to: {results['saved_txt_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
