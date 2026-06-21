from __future__ import annotations

from dataclasses import dataclass

from .models import ActionKind, AgentDecision, DecisionContext, GroupAssignment


@dataclass
class DecisionTools:
    agent_id: str
    context: DecisionContext

    def accept_assignment(self, contribution: float = 0.5, note: str = "") -> AgentDecision:
        return AgentDecision(kind=ActionKind.accept_assignment, accepted=True, contribution=contribution, note=note)

    def refuse_assignment(self, note: str = "") -> AgentDecision:
        return AgentDecision(kind=ActionKind.refuse_assignment, accepted=False, note=note)

    def contribute(self, contribution: float, note: str = "") -> AgentDecision:
        return AgentDecision(kind=ActionKind.contribute, contribution=contribution, note=note)

    def select_partners(self, target: str, note: str = "") -> AgentDecision:
        return AgentDecision(kind=ActionKind.select_partners, target=target, note=note)

    def move(self, target: str, note: str = "") -> AgentDecision:
        return AgentDecision(kind=ActionKind.move, target=target, note=note)

    def attack(self, target: str, note: str = "") -> AgentDecision:
        return AgentDecision(kind=ActionKind.attack, target=target, note=note)

    def wait(self, note: str = "") -> AgentDecision:
        return AgentDecision(kind=ActionKind.wait, note=note)

    def send_message(self, target: str, content: str, note: str = "") -> AgentDecision:
        return AgentDecision(kind=ActionKind.send_message, target=target, content=content, note=note)

    def build_prompt(self) -> str:
        memory_str = "\n".join(self.context.agent_memory) if self.context.agent_memory else "No memories yet."
        res_str = ", ".join(f"{k}: {int(v)}" for k, v in self.context.resources.items())
        
        nearby_agents_str = ", ".join(
            f"{other_id} (food: {self.context.other_agents_resources.get(other_id, 0)})"
            for other_id in self.context.nearby_agents
        ) if self.context.nearby_agents else "None"
        
        other_agents_str = ", ".join(
            f"{other_id} (location: {loc}, food: {self.context.other_agents_resources.get(other_id, 0)})"
            for other_id, loc in self.context.other_agents_locations.items()
        ) if self.context.other_agents_locations else "None"
        
        reachable_locs_str = ", ".join(self.context.reachable_locations) if self.context.reachable_locations else "None"
        
        beliefs_section = ""
        if self.context.personal_beliefs:
            beliefs_section = f"--- Personal Beliefs ---\n{self.context.personal_beliefs}\n\n"

        player_beliefs_str = "\n".join(f"- {name}: {belief}" for name, belief in self.context.player_beliefs.items()) if self.context.player_beliefs else "None yet."
        player_beliefs_section = f"--- Your Beliefs About Other Players ---\n{player_beliefs_str}\n\n"

        return (
            f"You are {self.agent_id}, an agent in a social simulation environment. Your goals include not starving and increasing your food.\n"
            f"Current Step: {self.context.step}\n"
            f"Your current location: {self.context.location}\n"
            f"Your current resources: {res_str}\n"
            f"Nearby agents: {nearby_agents_str}\n"
            f"Reachable locations: {reachable_locs_str}\n"
            f"Visible groups: {self.context.world_summary}\n"
            f"Other agents (locations & resources): {other_agents_str}\n\n"
            f"{beliefs_section}"
            f"{player_beliefs_section}"
            f"--- Core Rules & Mechanics ---\n"
            f"1. Resources & Survival: You have one symmetric resource (food). "
            f"Every round, you lose {self.context.depletion_per_round} unit(s) of food as cost of living. "
            f"Food must be contributed in integer quantities. "
            f"Watch out for starvation: if your food reaches 0 or below at the end of the round, you die of starvation (any player reaching 0 will die). "
            f"By design, not being able to increase your food by assignments for a long time means death by defection.\n"
            f"2. Group Assignment (Public Goods Game):\n"
            f"   - Every round, EVERY alive agent is assigned exactly one group of size 2. "
            f"Your partner is chosen randomly from all other alive agents, with probability PROPORTIONAL TO THEIR FOOD — richer agents are more likely to be chosen as partners. "
            f"Note that as you increase your resource you will be most likely to be assigned as a trading partner with others.\n"
            f"   - The group MULTIPLIER is shown in 'Visible groups' above and is known to BOTH members BEFORE they decide.\n"
            f"   - Each member independently decides how much food to contribute (integer, 0 to current food), or refuses. "
            f"If you refuse, you simply contribute 0 and are not penalized — but you forfeit any gain from the pool.\n"
            f"   - Payout formula: pool = floor(sum_of_contributions × multiplier), "
            f"then each ACCEPTING member receives floor(pool / num_acceptors). "
            f"IMPORTANT: results are rounded DOWN (floor) before distribution. "
            f"There is a round down logic: if you contribute like 1-1 (1 from each player) and the pool yields 3 food, you won't get any extra resource because the pool of 3 is rounded down to 2 when divided among 2 acceptors (3/2 rounded down to 1 each), leading back to a 1-1 split with zero net gain.\n"
            f"   - Example: both contribute 4, multiplier=1.7 → pool = floor(8×1.7) = floor(13.6) = 13 → each gets floor(13/2) = 6 food back.\n"
            f"   - If only 1 member accepts, effective multiplier = 1.0 (no gain from cooperating alone).\n"
            f"   - NOTE: You may appear as a partner in MULTIPLE groups this round (other agents may have selected you as their partner). Each group is resolved independently.\n"
            f"   - You MUST explain your contribution decision in the 'note' field.\n"
            f"3. Conflict & Attacks: You can 'attack(target)' a nearby agent. "
            f"20% chance you kill and loot all their food; 20% chance they defend and kill you; 60% chance draw (no effect). High risk. "
            f"You MUST explain your reasoning in the 'note' field.\n"
            f"4. Privacy: Your 'note' field and reasoning are PRIVATE. "
            f"Others only see your public actions (contribution amount, attack target) and direct messages.\n\n"
            f"--- Your Memory of Past Events ---\n"
            f"{memory_str}\n\n"
            f"Make a strategic decision based on your resources, your memories, and other agents. "
            f"Choose exactly one action using the available tools."
        )


class DecisionProvider:
    def decide(self, tools: DecisionTools) -> AgentDecision:
        raise NotImplementedError

    def decide_message(self, tools: DecisionTools) -> AgentDecision:
        return tools.wait(note="heuristic: skip message")

    def decide_evaluation(self, agent_id: str, context: DecisionContext, other_agents: list[str]) -> tuple[str, dict[str, str]]:
        evaluation = f"Default evaluation of step {context.step}."
        beliefs = {name: "No specific summary/belief." for name in other_agents}
        return evaluation, beliefs


class HeuristicDecisionProvider(DecisionProvider):
    def decide_evaluation(self, agent_id: str, context: DecisionContext, other_agents: list[str]) -> tuple[str, dict[str, str]]:
        evaluation = f"Heuristic evaluation of step {context.step}."
        beliefs = {name: "No specific summary/belief (heuristic)." for name in other_agents}
        return evaluation, beliefs
    """Baseline policy used until you plug in an actual LLM backend."""

    def decide(self, tools: DecisionTools) -> AgentDecision:
        context = tools.context
        decision = None
        
        if context.visible_groups:
            current_group = context.visible_groups[0]
            if context.agent_id in current_group.members:
                contribution = max(1, int(context.resources.get("food", 0) * 0.2))
                decision = tools.accept_assignment(contribution=contribution, note="baseline cooperation")

        if decision is None and context.nearby_agents:
            if context.step % 5 == 0:
                decision = tools.attack(target=context.nearby_agents[0], note="heuristic attack")
            else:
                decision = tools.select_partners(target=context.nearby_agents[0], note="choose nearest partner")

        if decision is None and context.reachable_locations:
            decision = tools.move(target=context.reachable_locations[0], note="explore neighboring location")

        if decision is None:
            decision = tools.wait(note="no salient action")

        decision.prompt = tools.build_prompt()
        decision.thinking = f"Executing heuristic choice: {decision.kind.value} targeting {decision.target}"

        from .llm import log_agent_decision
        log_agent_decision(tools.agent_id, context.step, decision.prompt, decision.thinking, decision)

        return decision


def build_decision_provider(config) -> DecisionProvider:
    if getattr(config.llm, "provider", "heuristic") == "heuristic":
        return HeuristicDecisionProvider()
    from .llm import ToolCallingLLMProvider

    return ToolCallingLLMProvider(
        model_name=config.llm.model_name,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        api_key_env=config.llm.api_key_env,
    )


def summarize_visible_groups(groups: list[GroupAssignment]) -> str:
    if not groups:
        return "No visible group assignments."
    return "; ".join(
        f"{group.group_id}: members={group.members}, multiplier={group.multiplier:.2f}, observers={group.observed_by}"
        for group in groups
    )
