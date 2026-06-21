from __future__ import annotations

from enum import Enum
from typing import Annotated

import networkx as nx
from pydantic import BaseModel, Field


class ResourceKind(str, Enum):
    food = "food"


class ActionKind(str, Enum):
    accept_assignment = "accept_assignment"
    refuse_assignment = "refuse_assignment"
    contribute = "contribute"
    select_partners = "select_partners"
    move = "move"
    attack = "attack"
    wait = "wait"
    send_message = "send_message"


class ResourceBundle(BaseModel):
    food: int = Field(default=0, ge=0)

    def as_dict(self) -> dict[str, int]:
        return self.model_dump()


class AgentMemoryEvent(BaseModel):
    step: int
    summary: str
    public: bool = False


class AgentState(BaseModel):
    agent_id: str
    position: str
    resources: ResourceBundle
    alive: bool = True
    memory: list[AgentMemoryEvent] = Field(default_factory=list)
    personal_beliefs: str = ""
    player_beliefs: dict[str, str] = Field(default_factory=dict)

    def record(self, event: AgentMemoryEvent, memory_size: int) -> None:
        self.memory.append(event)
        if memory_size <= 0:
            self.memory = []
        elif len(self.memory) > memory_size:
            self.memory = self.memory[-memory_size:]


class GroupAssignment(BaseModel):
    group_id: str
    members: list[str]
    multiplier: float
    observed_by: list[str] = Field(default_factory=list)


class AgentDecision(BaseModel):
    kind: ActionKind
    target: str | None = None
    contribution: int = 0
    accepted: bool = True
    note: str = ""
    content: str = ""
    prompt: str | None = None
    thinking: str | None = None


class StepReport(BaseModel):
    step: int
    public_events: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class WorldState(BaseModel):
    step: int = 0
    agents: dict[str, AgentState] = Field(default_factory=dict)
    groups: list[GroupAssignment] = Field(default_factory=list)
    geography: nx.Graph = Field(default_factory=nx.Graph, exclude=True)
    reports: list[StepReport] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class DecisionContext(BaseModel):
    step: int
    agent_id: str
    location: str
    nearby_agents: list[str] = Field(default_factory=list)
    reachable_locations: list[str] = Field(default_factory=list)
    visible_groups: list[GroupAssignment] = Field(default_factory=list)
    world_summary: str = ""
    resources: dict[str, float] = Field(default_factory=dict)
    agent_memory: list[str] = Field(default_factory=list)
    personal_beliefs: str = ""
    depletion_per_round: int = 1
    max_messages: int = 10
    other_agents_locations: dict[str, str] = Field(default_factory=dict)
    other_agents_resources: dict[str, int] = Field(default_factory=dict)
    player_beliefs: dict[str, str] = Field(default_factory=dict)
