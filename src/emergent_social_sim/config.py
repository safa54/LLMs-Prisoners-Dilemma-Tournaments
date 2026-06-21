from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GroupingConfig(BaseModel):
    max_group_size: int = Field(default=2, ge=2)
    resource_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_limit: int = Field(default=10, ge=1)
    assignment_probability: float = Field(default=0.6, ge=0.0, le=1.0)
    observer_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    public_goods_multiplier_min: float = Field(default=1.0, ge=1.0)
    public_goods_multiplier_max: float = Field(default=2.5, ge=1.0)


class PartnerSelectionConfig(BaseModel):
    resource_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_limit: int = Field(default=10, ge=1)


class ConflictConfig(BaseModel):
    attacker_kills_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    defender_kills_probability: float = Field(default=0.2, ge=0.0, le=1.0)


class GeographyConfig(BaseModel):
    nodes: list[str] = Field(default_factory=lambda: ["A"])
    edges: list[tuple[str, str]] = Field(default_factory=list)


class RewardConfig(BaseModel):
    cooperation_bonus: float = Field(default=0.25, ge=0.0)
    refusal_penalty: float = Field(default=0.1, ge=0.0)


class SimulationSection(BaseModel):
    steps: int = Field(default=20, ge=1)
    seed: int = Field(default=7)
    grouping: GroupingConfig = Field(default_factory=GroupingConfig)
    partner_selection: PartnerSelectionConfig = Field(default_factory=PartnerSelectionConfig)
    conflict: ConflictConfig = Field(default_factory=ConflictConfig)
    geography: GeographyConfig = Field(default_factory=GeographyConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)
    resources: dict[str, int] = Field(
        default_factory=lambda: {"food": 10}
    )
    depletion_per_round: int = Field(default=1, ge=0)
    max_messages: int = Field(default=10, ge=1)


class LLMSection(BaseModel):
    provider: str = Field(default="heuristic")
    model_name: str = Field(default="gpt-4.1-mini")
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    api_key_env: str = Field(default="OPENAI_API_KEY")
    reasoning_effort: str | None = Field(default=None)  # "low", "medium", "high" for o-series models


class AgentSection(BaseModel):
    count: int = Field(default=6, ge=2)
    memory_size: int = Field(default=12, ge=0)
    personal_beliefs: dict[str, str] = Field(default_factory=dict)
    starting_resources: dict[str, dict[str, int]] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    simulation: SimulationSection = Field(default_factory=SimulationSection)
    agents: AgentSection = Field(default_factory=AgentSection)
    llm: LLMSection = Field(default_factory=LLMSection)

    @classmethod
    def from_toml(cls, path: str | Path) -> "SimulationConfig":
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
            try:
                import tomli as tomllib
            except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
                raise RuntimeError("TOML support requires Python 3.11+ or the optional 'tomli' package.") from error

        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
