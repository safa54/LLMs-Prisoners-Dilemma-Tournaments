from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class PayoutMatrix(BaseModel):
    T: int = Field(default=5, description="Temptation payoff (defect while other cooperates)")
    R: int = Field(default=3, description="Reward payoff (both cooperate)")
    P: int = Field(default=1, description="Punishment payoff (both defect)")
    S: int = Field(default=0, description="Sucker's payoff (cooperate while other defects)")

    @model_validator(mode="after")
    def validate_payoffs(self) -> PayoutMatrix:
        if not (self.T > self.R > self.P > self.S):
            raise ValueError(f"Payoffs must satisfy T > R > P > S. Got: T={self.T}, R={self.R}, P={self.P}, S={self.S}")
        if 2 * self.R <= self.T + self.S:
            raise ValueError(f"Payoffs must satisfy 2*R > T + S for classic Prisoner's Dilemma. Got: 2*R={2*self.R}, T+S={self.T + self.S}")
        return self


class GameConfig(BaseModel):
    rounds: int = Field(default=10, ge=1, description="Number of rounds in each 2-player game")
    payouts: PayoutMatrix = Field(default_factory=PayoutMatrix)
    enable_communication: bool = Field(default=False, description="Enable one pre-game message exchange at the start of a match")


class AgentConfig(BaseModel):
    agent_id: str = Field(..., description="Unique name/identifier for the agent")
    agent_type: Literal["llm", "bot"] = Field(..., description="Whether the agent is an LLM or deterministic bot")
    
    # LLM-specific fields
    model_name: str | None = Field(default=None, description="LiteLLM model identifier (e.g. gpt-4o, claude-3-5-sonnet, gemini/gemini-1.5-pro)")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    reasoning_effort: str | None = Field(default=None, description="Optional reasoning effort (low, medium, high) for o-series models")
    system_prompt: str | None = Field(default=None, description="Personality, goal guidelines or custom guidelines for the LLM agent")
    api_key: str | None = Field(default=None, description="Optional API key for this agent's model")
    
    # Bot-specific fields
    bot_type: str | None = Field(default=None, description="Type of deterministic bot (e.g., 'always_cooperate')")


class TournamentConfig(BaseModel):
    game: GameConfig = Field(default_factory=GameConfig)
    agents: list[AgentConfig] = Field(default_factory=list)
    log_file: str | None = Field(default="logs/simple_pd_tournament.json", description="Path to write the logs of the tournament")
    match_cache_file: str | None = Field(default="logs/match_cache.json", description="Path to the persistent match cache file")

    @classmethod
    def from_json(cls, path: str | Path) -> TournamentConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TournamentConfig:
        return cls.model_validate(data)
