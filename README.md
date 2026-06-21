# LLMs Iterated Prisoner's Dilemma Tournaments

This repository contains the codebase and configuration files for running and analyzing Iterated Prisoner's Dilemma (IPD) tournaments between Large Language Model (LLM) agents and classical rule-based strategies. The project investigates how strategic adaptations (reputation, elimination/survival threat, and pre-game communication) influence LLM cooperation rates.

Additionally, this repository includes an experimental scaffold for study of emergent social norms among LLM agents in a graph-based world using LangGraph (`src/emergent_social_sim`).

---

## 🎮 Prisoner's Dilemma Tournament (simple_pd)

The main simulation framework is structured as a round-robin tournament where every agent plays a 20-round match against every other agent in the pool.

### Strategies Included
1. **Rule-Based Bots (`bot`)**:
   - `AlwaysCoopBot`: Unconditional cooperation.
   - `AlwaysDefectBot`: Unconditional defection.
   - `TitForTatBot`: Cooperates on round 1, then mimics the opponent's previous move.
   - `ForgivingTFT`: Tit-for-Tat but occasionally forgives defection to avoid retaliation spirals.
   - `ManipulativeTFT`: Cooperates initially, then defects periodically to exploit cooperation.
   - `TitForTwoTats`: Defects only after two consecutive defections.
   - `RandomBot`: Cooperates or defects randomly with equal probability.

2. **LLM Agents (`llm`)**:
   - `GPT-5.4-Mini` (lite reasoning agent)
   - `GPT-5.5` (large reasoning agent)
   - `Gemini 3.5 Flash` (fast generalist agent)
   - `Claude Sonnet 4.6` (reasoning-oriented agent)

LLM agents are queried via LiteLLM and prompted to produce a structured JSON response containing their action choice (Cooperate or Defect) and their strategic reasoning.

### Experimental Scenarios (`configs/`)
* **Scenario A: Baseline IPD** (`configs/simple_pd.json`): Agents are instructed to maximize their own cumulative score.
* **Scenario B: Perceived Reputation** (`configs/simple_pd_fake_reputation.json`): Prompt framing states that choice history is visible to future opponents (perceived reputation).
* **Scenario C: Perceived Starvation/Elimination** (`configs/simple_pd_fake_starvation.json`): Prompt framing states that low scorers will be eliminated from the opponent pool.
* **Scenario D: Pre-Game Communication** (`configs/simple_pd_communication.json`): Runs an LLM-only subpool where agents exchange a single 100-character pre-game message before the first round.

---

## 🛠️ Project Structure

- `src/simple_pd/cli.py` – CLI entry point to configure and run tournaments.
- `src/simple_pd/tournament.py` – Tournament orchestrator (handles matchups, standings, log writing).
- `src/simple_pd/game.py` – Iterated Prisoner's Dilemma round runner and payout matrices.
- `src/simple_pd/agent.py` – Implementation of heuristic bots and LiteLLM prompt adapters.
- `src/simple_pd/config.py` – Pydantic models for configuration parsing.
- `configs/` – JSON scenario configs for different experimental runs.
- `analyze_subpools.py` – Analysis utility to recalculate subgroup standings (e.g. LLM-only, LLM + cooperative bots, LLM + exploitative bots).
- `interpret_results.py` – Utility to parse tournament JSON output logs and generate structured TXT reports of model reasoning patterns.
- `main.tex` – LaTeX source for the academic report documenting tournament findings, subgroup standings, and pairwise matchup matrices.

---

## 🚀 Getting Started

### Installation
Clone the repository and install it in editable mode:
```bash
git clone https://github.com/safa54/LLMs-Prisoners-Dilemma-Tournaments.git
cd LLMs-Prisoners-Dilemma-Tournaments
pip install -e .
```

### Environment Configuration
The tournament relies on LiteLLM. Set the appropriate environment variables for the API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

### Running a Tournament
Run a tournament scenario by pointing the CLI to a configuration file:
```bash
python -m simple_pd.cli --config configs/simple_pd.json
```
The output logs (both JSON and human-readable TXT) are saved in the `logs/` directory.

### Running Analysis Scripts
1. **Recalculate Subgroup Standings**:
   ```bash
   python analyze_subpools.py -i logs/simple_pd_tournament_xxxxxx.json
   ```
2. **Interpret LLM Reasoning Patterns**:
   ```bash
   python interpret_results.py -i logs/simple_pd_tournament_xxxxxx.json
   ```

---

## 🗺️ Emergent Social Simulation (emergent_social_sim)

An auxiliary workflow designed to study multi-agent coordination. It assigns agents to physical locations on a graph and runs LangGraph orchestration loops to simulate resource depletion, public-goods sharing, partner choices, migration, and conflicts.

### Run Emergent Simulation
```bash
python -m emergent_social_sim.cli --config configs/default.json --steps 20
```
