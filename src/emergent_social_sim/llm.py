from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .models import AgentDecision
from .policy import DecisionTools


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class LLMResponder(Protocol):
    def invoke(self, prompt: str, tools: list[ToolCall]) -> AgentDecision:
        raise NotImplementedError


RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_decision_logs_buffer: list[str] = []


def log_raw_text(text: str) -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_txt = logs_dir / f"simulation_run_{RUN_TIMESTAMP}.txt"
    with open(log_file_txt, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def log_agent_decision(agent_id: str, step: int, prompt: str | None, thinking: str | None, decision: AgentDecision) -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSONL log (appended across all runs)
    log_file_jsonl = logs_dir / "simulation_prompt_log.jsonl"
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "agent_id": agent_id,
        "prompt": prompt,
        "thinking": thinking,
        "decision": decision.model_dump()
    }
    with open(log_file_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    # 2. Buffer Text log details in-memory instead of writing to simulation_run_{RUN_TIMESTAMP}.txt immediately
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append(f"TIMESTAMP: {datetime.datetime.now().isoformat()}")
    log_lines.append(f"STEP: {step}")
    log_lines.append(f"AGENT: {agent_id}")
    log_lines.append("=" * 80)
    log_lines.append("--- AGENT THINKING ---")
    log_lines.append(f"{thinking or 'None'}\n")
    log_lines.append("--- AGENT DECISION ---")
    log_lines.append(f"Action Kind: {decision.kind.value}")
    if decision.target:
        log_lines.append(f"Target:      {decision.target}")
    if decision.kind.value == "send_message":
        log_lines.append(f"Content:     {decision.content}")
    if decision.contribution:
        log_lines.append(f"Contribution: {decision.contribution}")
    log_lines.append(f"Accepted:     {decision.accepted}")
    if decision.note:
        log_lines.append(f"Note:         {decision.note}")
    log_lines.append("\n\n")

    _decision_logs_buffer.append("\n".join(log_lines))


def flush_agent_decisions() -> None:
    global _decision_logs_buffer
    if not _decision_logs_buffer:
        return
    
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_txt = logs_dir / f"simulation_run_{RUN_TIMESTAMP}.txt"
    
    header = (
        "\n"
        "================================================================================\n"
        "DETAILED AGENT DECISIONS (PROMPTS & THINKING) FOR THIS ROUND\n"
        "================================================================================\n\n"
    )
    with open(log_file_txt, "a", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(_decision_logs_buffer))
    _decision_logs_buffer.clear()


def log_round_details(
    step: int,
    groups: list,
    public_events: list[str],
    conflicts: list[str],
    agents: dict,
) -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_txt = logs_dir / f"simulation_run_{RUN_TIMESTAMP}.txt"
    with open(log_file_txt, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ROUND RESOLUTION: STEP {step}\n")
        f.write("=" * 80 + "\n")
        
        # 1. Log assignments
        f.write("--- GROUP ASSIGNMENTS ---\n")
        if groups:
            for g in groups:
                f.write(f"Group: {g.group_id} | Members: {', '.join(g.members)} | Multiplier: {g.multiplier:.2f}\n")
        else:
            f.write("No groups assigned this step.\n")
        f.write("\n")
        
        # 2. Log events and actions
        f.write("--- ROUND EVENTS & ACTIONS ---\n")
        all_events = public_events + conflicts
        if all_events:
            for ev in all_events:
                f.write(f"- {ev}\n")
        else:
            f.write("No major public events or actions occurred.\n")
        f.write("\n")
        
        # 3. Log resource distributions and balances
        f.write("--- AGENT RESOURCES & BALANCE (POST-INCOME) ---\n")
        f.write("Income distribution: No food resources awarded this round.\n")
        for agent_id, agent in agents.items():
            status = "ALIVE" if agent.alive else "DEAD"
            res = agent.resources
            res_str = f"food: {res.food}"
            f.write(f"Agent: {agent_id:<8} | Status: {status:<5} | Resources: {res_str}\n")
        f.write("\n\n")


def log_system_summary(step: int, prompt: str, summary: str) -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSONL log
    log_file_jsonl = logs_dir / "simulation_prompt_log.jsonl"
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "agent_id": "SYSTEM_SUMMARIZER",
        "prompt": prompt,
        "thinking": "Generating global rounds summary",
        "summary": summary
    }
    with open(log_file_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    # 2. Text log
    log_file_txt = logs_dir / f"simulation_run_{RUN_TIMESTAMP}.txt"
    with open(log_file_txt, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"SYSTEM SUMMARIZER (Step {step})\n")
        f.write("=" * 80 + "\n")
        f.write("--- SUMMARIZER PROMPT ---\n")
        f.write(f"{prompt}\n\n")
        f.write("--- GENERATED ROUNDS SUMMARY ---\n")
        f.write(f"{summary}\n")
        f.write("\n\n")


def generate_summary(config: SimulationConfig, events_text: str) -> str:
    prompt = f"Summarize what happened throughout these simulation rounds based on the following logs:\n\n{events_text}"
    if config.llm.provider == "heuristic":
        return f"[Heuristic Summary]: Simulated events summary for prompt: '{prompt[:60]}...'"
    
    try:
        import os
        api_key_env_val = config.llm.api_key_env
        api_key = os.environ.get(api_key_env_val)
        if not api_key:
            if api_key_env_val and api_key_env_val.startswith("sk-"):
                api_key = api_key_env_val
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
        
        # Check for OpenAI
        import openai
        client = openai.OpenAI(api_key=api_key)
        kwargs = {
            "model": config.llm.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.llm.temperature,
        }
        if getattr(config.llm, "reasoning_effort", None):
            kwargs["reasoning_effort"] = config.llm.reasoning_effort
        try:
            response = client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if "temperature" in str(e).lower() and "temperature" in kwargs:
                kwargs.pop("temperature")
                response = client.chat.completions.create(**kwargs)
            else:
                raise e
        return response.choices[0].message.content or "No summary generated."
    except Exception as e:
        return f"[Summary generation error: {e}]. Please install 'openai' and configure the environment variable to get LLM-generated summaries."


def get_openai_tools_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "accept_assignment",
                "description": "Accept the group assignment and contribute resources to the public goods pool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contribution": {
                            "type": "integer",
                            "description": "Amount of resources to contribute (an integer between 0 and your current food)."
                        },
                        "note": {
                            "type": "string",
                            "description": "Reasoning why you decided to contribute this specific amount."
                        }
                    },
                    "required": ["contribution", "note"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "refuse_assignment",
                "description": "Refuse the group assignment. A penalty resource may apply.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Explanation or note about this choice."
                        }
                    },
                    "required": ["note"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "contribute",
                "description": "Contribute a specific amount of resources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contribution": {
                            "type": "integer",
                            "description": "Amount to contribute (an integer)."
                        },
                        "note": {
                            "type": "string",
                            "description": "Reasoning why you decided to contribute this specific amount."
                        }
                    },
                    "required": ["contribution", "note"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "select_partners",
                "description": "Propose to cooperate bilaterally with a specific nearby agent. Cooperation only succeeds if mutual.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Name of the target agent to cooperate with."
                        },
                        "note": {
                            "type": "string",
                            "description": "Explanation or note about this choice."
                        }
                    },
                    "required": ["target", "note"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move to a neighboring location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Neighbor node/location to move to."
                        },
                        "note": {
                            "type": "string",
                            "description": "Explanation or note about this choice."
                        }
                    },
                    "required": ["target", "note"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "attack",
                "description": "Deliberately attack another nearby agent at your location. High risk of death or looting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Name of the nearby agent to attack."
                        },
                        "note": {
                            "type": "string",
                            "description": "Reasoning why you decided to attack this specific agent."
                        }
                    },
                    "required": ["target", "note"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wait",
                "description": "Wait and do nothing this turn.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Explanation or note about this choice."
                        }
                    },
                    "required": ["note"]
                }
            }
        }
    ]


def get_openai_message_tools_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a brief message (max 200 characters) to another agent at your location. Communicate how much you want to contribute to the public goods pot or coordinate partnerships.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Name of the nearby agent to message."
                        },
                        "content": {
                            "type": "string",
                            "description": "Message content (max 200 characters)."
                        },
                        "note": {
                            "type": "string",
                            "description": "Your internal explanation for choosing to send this message."
                        }
                    },
                    "required": ["target", "content", "note"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wait",
                "description": "Skip sending a message or finish messaging this round.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Explanation for skipping messaging."
                        }
                    },
                    "required": ["note"]
                }
            }
        }
    ]



# ---------------------------------------------------------------------------
# Responses-API tool schemas (flat format, no nested "function" key)
# ---------------------------------------------------------------------------

def get_responses_tools_schema() -> list[dict]:
    return [
        {"type":"function","name":"accept_assignment",
         "description":"Accept the group assignment and contribute resources to the public goods pool.",
         "parameters":{"type":"object","properties":{
             "contribution":{"type":"integer","description":"Amount of food to contribute (0 to your current food)."},
             "note":{"type":"string","description":"Reasoning why you decided to contribute this specific amount."}},
             "required":["contribution","note"]}},
        {"type":"function","name":"refuse_assignment",
         "description":"Refuse the group assignment.",
         "parameters":{"type":"object","properties":{
             "note":{"type":"string","description":"Explanation for refusing."}},
             "required":["note"]}},
        {"type":"function","name":"contribute",
         "description":"Contribute a specific amount of resources.",
         "parameters":{"type":"object","properties":{
             "contribution":{"type":"integer","description":"Amount to contribute."},
             "note":{"type":"string","description":"Reasoning why you decided to contribute this specific amount."}},
             "required":["contribution","note"]}},
        {"type":"function","name":"select_partners",
         "description":"Propose bilateral cooperation with a specific nearby agent.",
         "parameters":{"type":"object","properties":{
             "target":{"type":"string","description":"Name of the target agent."},
             "note":{"type":"string","description":"Explanation for this choice."}},
             "required":["target","note"]}},
        {"type":"function","name":"move",
         "description":"Move to a neighboring location.",
         "parameters":{"type":"object","properties":{
             "target":{"type":"string","description":"Neighbor node to move to."},
             "note":{"type":"string","description":"Explanation for this choice."}},
             "required":["target","note"]}},
        {"type":"function","name":"attack",
         "description":"Deliberately attack another nearby agent. 20% kill, 20% you die, 60% draw.",
         "parameters":{"type":"object","properties":{
             "target":{"type":"string","description":"Name of the nearby agent to attack."},
             "note":{"type":"string","description":"Reasoning why you decided to attack this agent."}},
             "required":["target","note"]}},
        {"type":"function","name":"wait",
         "description":"Wait and do nothing this turn.",
         "parameters":{"type":"object","properties":{
             "note":{"type":"string","description":"Explanation for waiting."}},
             "required":["note"]}},
    ]


def get_responses_message_tools_schema() -> list[dict]:
    return [
        {"type":"function","name":"send_message",
         "description":"Send a brief message (max 200 characters) to another agent.",
         "parameters":{"type":"object","properties":{
             "target":{"type":"string","description":"Name of the nearby agent to message."},
             "content":{"type":"string","description":"Message content (max 200 characters)."},
             "note":{"type":"string","description":"Your internal explanation (private)."}},
             "required":["target","content","note"]}},
        {"type":"function","name":"wait",
         "description":"Skip sending a message.",
         "parameters":{"type":"object","properties":{
             "note":{"type":"string","description":"Explanation for skipping."}},
             "required":["note"]}},
    ]


def get_responses_evaluation_tools_schema(other_agents: list[str]) -> list[dict]:
    return [
        {"type":"function","name":"submit_evaluation",
         "description":"Submit your round evaluation and updated player beliefs.",
         "parameters":{"type":"object","properties":{
             "evaluation":{"type":"string","description":"Your evaluation of the round, plan and insights."},
             "player_beliefs":{"type":"object",
                 "description":"Map from agent name to your updated belief about them.",
                 "properties":{name:{"type":"string","description":f"Updated belief about {name}."} for name in other_agents},
                 "required":other_agents}},
             "required":["evaluation","player_beliefs"]}},
    ]


@dataclass
class ToolCallingLLMProvider:
    """Adapter supporting both OpenAI Chat Completions (gpt-4.x, o-series)
    and the new Responses API (gpt-5.x).

    - gpt-5.x  -> client.responses.create  with reasoning={"effort": ...}
    - o1/o3/o4 -> client.chat.completions  with reasoning_effort=
    - gpt-4.x  -> client.chat.completions  with temperature=
    """

    model_name: str
    temperature: float = 0.5
    max_tokens: int = 8000
    api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str | None = None  # "low", "medium", "high"

    def _get_client(self):
        import os
        import openai
        api_key = os.environ.get(self.api_key_env)
        if not api_key and self.api_key_env.startswith("sk-"):
            api_key = self.api_key_env
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OpenAI API key not found. Set '{self.api_key_env}' environment variable.")
        return openai.OpenAI(api_key=api_key)

    def _use_responses_api(self) -> bool:
        """gpt-5.x models use the new Responses API."""
        return self.model_name.lower().startswith("gpt-5")

    def _is_o_series(self) -> bool:
        name = self.model_name.lower()
        return name.startswith("o1") or name.startswith("o3") or name.startswith("o4")

    # ------------------------------------------------------------------
    # Responses API path (gpt-5.x)
    # ------------------------------------------------------------------

    def _call_responses_api(
        self,
        prompt: str,
        tools: list[dict],
        tool_choice: str | dict = "required",
    ) -> tuple[str, str | None, dict | None]:
        """Returns (thinking, tool_name, tool_args)."""
        import json
        client = self._get_client()

        kwargs: dict = {
            "model": self.model_name,
            "input": [{"role": "user", "content": prompt}],
            "tools": tools,
            "tool_choice": tool_choice,
            "max_output_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        response = client.responses.create(**kwargs)

        thinking_parts: list[str] = []
        tool_name: str | None = None
        tool_args: dict | None = None

        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                for s in (getattr(item, "summary", []) or []):
                    text = getattr(s, "text", None) or (s if isinstance(s, str) else "")
                    if text:
                        thinking_parts.append(text)
            elif item_type == "function_call":
                tool_name = item.name
                raw = item.arguments
                tool_args = raw if isinstance(raw, dict) else json.loads(raw)
            elif item_type == "message":
                for block in (getattr(item, "content", []) or []):
                    text = getattr(block, "text", None)
                    if text:
                        thinking_parts.append(text)

        return "\n".join(thinking_parts), tool_name, tool_args

    # ------------------------------------------------------------------
    # Chat Completions path (gpt-4.x, o-series)
    # ------------------------------------------------------------------

    def _call_chat_completions(
        self,
        prompt: str,
        tools: list[dict],
        tool_choice: str | dict = "required",
    ) -> tuple[str, str | None, dict | None]:
        """Returns (thinking, tool_name, tool_args)."""
        import json
        import openai
        client = self._get_client()

        kwargs: dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
            "tool_choice": tool_choice,
        }

        if self._is_o_series():
            kwargs["max_completion_tokens"] = self.max_tokens
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature

        try:
            response = client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            err = str(e).lower()
            if "temperature" in err:
                kwargs.pop("temperature", None)
            if "max_tokens" in err or "unsupported_parameter" in err:
                t = kwargs.pop("max_tokens", None)
                if t:
                    kwargs["max_completion_tokens"] = t
            response = client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        thinking = ""
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            thinking = message.reasoning_content
        elif message.content:
            thinking = message.content

        if not message.tool_calls:
            return thinking, None, None

        tc = message.tool_calls[0]
        return thinking, tc.function.name, json.loads(tc.function.arguments)

    # ------------------------------------------------------------------
    # Unified dispatch
    # ------------------------------------------------------------------

    def _call(
        self,
        prompt: str,
        chat_tools: list[dict],
        responses_tools: list[dict],
        tool_choice: str | dict = "required",
    ) -> tuple[str, str | None, dict | None]:
        if self._use_responses_api():
            return self._call_responses_api(prompt, responses_tools, tool_choice)
        else:
            return self._call_chat_completions(prompt, chat_tools, tool_choice)

    def _parse_action(self, thinking: str, tool_name: str | None, tool_args: dict | None) -> AgentDecision:
        from .models import ActionKind
        if tool_name is None or tool_args is None:
            return AgentDecision(kind=ActionKind.wait, thinking=thinking, note="No tool called by LLM")
        contribution_val = tool_args.get("contribution", 0)
        try:
            contribution = int(float(contribution_val))
        except (ValueError, TypeError):
            contribution = 0
        try:
            kind = ActionKind(tool_name)
        except ValueError:
            kind = ActionKind.wait
        return AgentDecision(
            kind=kind,
            target=tool_args.get("target", None),
            contribution=contribution,
            accepted=tool_args.get("accepted", True),
            note=tool_args.get("note", ""),
            content=tool_args.get("content", ""),
            thinking=thinking,
        )

    def invoke(self, prompt: str, tools: list[ToolCall], custom_schema: list[dict] | None = None) -> AgentDecision:
        """Legacy entry point."""
        chat_schema = custom_schema if custom_schema is not None else get_openai_tools_schema()
        is_msg = custom_schema is not None and any(
            t.get("function", {}).get("name") == "send_message" for t in custom_schema
        )
        resp_schema = get_responses_message_tools_schema() if is_msg else get_responses_tools_schema()
        thinking, tool_name, tool_args = self._call(prompt, chat_schema, resp_schema)
        return self._parse_action(thinking, tool_name, tool_args)

    def decide(self, tools: DecisionTools) -> AgentDecision:
        prompt = tools.build_prompt()
        thinking, tool_name, tool_args = self._call(
            prompt, get_openai_tools_schema(), get_responses_tools_schema()
        )
        decision = self._parse_action(thinking, tool_name, tool_args)
        decision.prompt = prompt
        log_agent_decision(tools.agent_id, tools.context.step, decision.prompt, decision.thinking, decision)
        return decision

    def decide_message(self, tools: DecisionTools) -> AgentDecision:
        memory_str = "\n".join(tools.context.agent_memory) if tools.context.agent_memory else "No memories yet."
        res_str = ", ".join(f"{k}: {int(v)}" for k, v in tools.context.resources.items())
        nearby_agents_str = ", ".join(
            f"{other_id} (food: {tools.context.other_agents_resources.get(other_id, 0)})"
            for other_id in tools.context.nearby_agents
        ) if tools.context.nearby_agents else "None"
        beliefs_section = f"--- Personal Beliefs ---\n{tools.context.personal_beliefs}\n\n" if tools.context.personal_beliefs else ""
        player_beliefs_str = "\n".join(f"- {name}: {belief}" for name, belief in tools.context.player_beliefs.items()) if tools.context.player_beliefs else "None yet."

        prompt = (
            f"You are {tools.agent_id}, in the communication phase of step {tools.context.step}.\n"
            f"Your current location: {tools.context.location}\n"
            f"Your current resources: {res_str}\n"
            f"Nearby agents: {nearby_agents_str}\n"
            f"Visible groups: {tools.context.world_summary}\n\n"
            f"{beliefs_section}"
            f"--- Your Beliefs About Other Players ---\n{player_beliefs_str}\n\n"
            f"--- Communication Phase Rules ---\n"
            f"1. Use this phase to communicate with other agents. "
            f"Every agent this round is assigned exactly one group of size 2 (partner chosen proportional to food). "
            f"Group assignments shown under 'Visible groups'. Coordinate contribution levels with your partner. "
            f"Every round you lose {tools.context.depletion_per_round} food. Reaching 0 means death.\n"
            f"2. Your message must be brief (maximum 200 characters).\n"
            f"3. You can send at most {tools.context.max_messages} messages this round. Choose 'wait' when done.\n"
            f"4. Privacy: Your 'note' field is private. Only 'content' is seen by the recipient.\n\n"
            f"--- Your Memory of Past Events ---\n"
            f"{memory_str}\n\n"
            f"Choose whether to 'send_message(target, content)' or 'wait'."
        )

        thinking, tool_name, tool_args = self._call(
            prompt, get_openai_message_tools_schema(), get_responses_message_tools_schema()
        )
        decision = self._parse_action(thinking, tool_name, tool_args)
        decision.prompt = prompt
        log_agent_decision(tools.agent_id, tools.context.step, decision.prompt, decision.thinking, decision)
        return decision

    def decide_evaluation(self, agent_id: str, context: DecisionContext, other_agents: list[str]) -> tuple[str, dict[str, str]]:
        prompt = build_evaluation_prompt(agent_id, context, other_agents)

        chat_choice: str | dict = {"type": "function", "function": {"name": "submit_evaluation"}}
        resp_choice: str | dict = {"type": "function", "name": "submit_evaluation"}

        thinking, tool_name, tool_args = self._call(
            prompt,
            get_openai_evaluation_tools_schema(other_agents),
            get_responses_evaluation_tools_schema(other_agents),
            tool_choice=resp_choice if self._use_responses_api() else chat_choice,
        )

        if tool_args is None:
            return "No evaluation submitted.", {name: "No belief submitted." for name in other_agents}

        evaluation = tool_args.get("evaluation", "No evaluation submitted.")
        beliefs_raw = tool_args.get("player_beliefs", {})
        cleaned_beliefs = {name: str(beliefs_raw.get(name, "No specific belief yet.")) for name in other_agents}
        return evaluation, cleaned_beliefs


def build_evaluation_prompt(agent_id: str, context: DecisionContext, other_agents: list[str]) -> str:
    memory_str = "\n".join(context.agent_memory) if context.agent_memory else "No memories yet."
    beliefs_str = "\n".join(f"- {name}: {belief}" for name, belief in context.player_beliefs.items()) if context.player_beliefs else "None."
    
    return (
        f"You are {agent_id}, evaluating step {context.step}.\n"
        f"Your current resources: {context.resources}\n"
        f"Your current location: {context.location}\n\n"
        f"--- Your Memory of Past Events ---\n"
        f"{memory_str}\n\n"
        f"--- Your Previous Beliefs About Other Players ---\n"
        f"{beliefs_str}\n\n"
        f"Task:\n"
        f"1. Evaluate this round: reflect on what happened, check if your plan succeeded, and detail your plan and insights for the next round.\n"
        f"2. Update your summaries/beliefs about each other player in the game: reflect on their actions, messages, and cooperation this round, and write an updated brief summary/belief for each player.\n\n"
        f"Response Format:\n"
        f"You MUST call 'submit_evaluation' with your evaluation of the round and your updated beliefs/summaries about other players."
    )


def get_openai_evaluation_tools_schema(other_agents: list[str]) -> list[dict]:
    properties = {
        "evaluation": {
            "type": "string",
            "description": "Your evaluation about the round, your plan and insights."
        },
        "player_beliefs": {
            "type": "object",
            "description": "An object mapping agent names to your updated summary/belief about them.",
            "properties": {
                name: {
                    "type": "string",
                    "description": f"Updated summary/belief about {name}."
                }
                for name in other_agents
            },
            "required": other_agents
        }
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_evaluation",
                "description": "Submit your round evaluation and player beliefs.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": ["evaluation", "player_beliefs"]
                }
            }
        }
    ]

