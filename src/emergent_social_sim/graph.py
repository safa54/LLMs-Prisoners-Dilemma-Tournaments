from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - optional runtime dependency
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - fallback for bare environments
    END = "__end__"
    StateGraph = None

from .config import SimulationConfig
from .models import WorldState
from .policy import DecisionProvider
from .simulation import advance_world


def build_simulation_graph(config: SimulationConfig, provider: DecisionProvider):
    if StateGraph is None:
        return _FallbackGraph(config, provider)

    graph = StateGraph(WorldState)

    def step_node(world: WorldState) -> WorldState:
        import random

        rng = random.Random(config.simulation.seed + world.step)
        return advance_world(world, config, provider, rng)

    graph.add_node("step", step_node)
    graph.set_entry_point("step")
    graph.add_edge("step", END)
    return graph.compile()


@dataclass
class _FallbackGraph:
    config: SimulationConfig
    provider: DecisionProvider

    def invoke(self, world: WorldState) -> WorldState:
        import random

        rng = random.Random(self.config.simulation.seed + world.step)
        return advance_world(world, self.config, self.provider, rng)
