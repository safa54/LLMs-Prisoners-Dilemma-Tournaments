from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SimulationConfig
from .graph import build_simulation_graph
from .policy import build_decision_provider
from .simulation import create_world
from .llm import RUN_TIMESTAMP


def load_config(path: str | None) -> SimulationConfig:
    if path is None:
        default_path = Path("configs/default.json")
        if default_path.exists():
            return SimulationConfig.from_json(default_path)
        return SimulationConfig()
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        return SimulationConfig.from_json(config_path)
    return SimulationConfig.from_toml(config_path)


def run_simulation(config: SimulationConfig):
    provider = build_decision_provider(config)
    graph = build_simulation_graph(config, provider)
    world = create_world(config)

    for _ in range(config.simulation.steps):
        world = graph.invoke(world)

    return world


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the emergent social simulation.")
    parser.add_argument("--config", type=str, default=None, help="Path to a TOML configuration file.")
    parser.add_argument("--steps", type=int, default=None, help="Override the configured number of steps.")
    args = parser.parse_args()

    config = load_config(args.config)
    
    from .llm import log_raw_text

    # Helper function to print summary of a state to the txt log
    def print_summary(w, reports_to_show):
        summary = {
            "step": w.step,
            "agents": {
                agent_id: {
                    "position": agent.position,
                    "resources": agent.resources.as_dict(),
                    "alive": agent.alive,
                }
                for agent_id, agent in w.agents.items()
            },
            "reports": [report.model_dump() for report in reports_to_show],
        }
        log_raw_text(json.dumps(summary, indent=2, sort_keys=True))

    def generate_and_log_round_summary(w, reports_slice):
        if not reports_slice:
            return
        
        lines = []
        for r in reports_slice:
            lines.append(f"Step {r.step}:")
            for ev in r.public_events:
                lines.append(f"  - {ev}")
            for conf in r.conflicts:
                lines.append(f"  - Conflict: {conf}")
        events_text = "\n".join(lines)
        
        from .llm import generate_summary, log_system_summary
        
        log_raw_text("\n--- Generating LLM Summary of Rounds ---")
        summary_text = generate_summary(config, events_text)
        log_raw_text(summary_text)
        
        log_system_summary(w.step, f"Summarize logs:\n{events_text}", summary_text)

    # Non-interactive CLI override
    if args.steps is not None:
        provider = build_decision_provider(config)
        graph = build_simulation_graph(config, provider)
        world = create_world(config)
        for _ in range(args.steps):
            world = graph.invoke(world)
        print_summary(world, world.reports)
        generate_and_log_round_summary(world, world.reports)
        return

    # Interactive Loop
    provider = build_decision_provider(config)
    graph = build_simulation_graph(config, provider)
    world = create_world(config)

    log_raw_text("Welcome to the Emergent Social Simulation!")
    log_raw_text(f"World initialized with {len(world.agents)} agents.")
    log_raw_text("\nInitial State:")
    print_summary(world, [])

    print("Welcome to the Emergent Social Simulation!")
    print(f"World initialized with {len(world.agents)} agents. Logs saving to logs/simulation_run_{RUN_TIMESTAMP}.txt.")

    while True:
        try:
            user_input = input("\nEnter number of steps to run (or 'q' to quit): ").strip()
            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting simulation.")
                break
            if not user_input:
                continue
            
            steps = int(user_input)
            if steps <= 0:
                print("Please enter a positive integer.")
                continue
            
            print(f"Running {steps} steps... (Logging details to logs/simulation_run_{RUN_TIMESTAMP}.txt)")
            log_raw_text(f"\nRunning {steps} steps...")
            
            start_report_idx = len(world.reports)
            for _ in range(steps):
                world = graph.invoke(world)
            
            log_raw_text("\n--- Run Summary ---")
            reports_to_show = world.reports[start_report_idx:]
            print_summary(world, reports_to_show)
            generate_and_log_round_summary(world, reports_to_show)
            print("Done.")

        except ValueError:
            print("Invalid input. Please enter an integer or 'q' to quit.")
        except KeyboardInterrupt:
            print("\nExiting simulation.")
            break


if __name__ == "__main__":
    main()
