from __future__ import annotations

import math
import random
from collections.abc import Callable

import networkx as nx

from .config import SimulationConfig
from .models import (
    ActionKind,
    AgentDecision,
    AgentMemoryEvent,
    AgentState,
    DecisionContext,
    GroupAssignment,
    ResourceBundle,
    StepReport,
    WorldState,
)
from .policy import DecisionProvider, DecisionTools, summarize_visible_groups


def create_world(config: SimulationConfig) -> WorldState:
    graph = nx.Graph()
    graph.add_nodes_from(config.simulation.geography.nodes)
    graph.add_edges_from(config.simulation.geography.edges)

    agents: dict[str, AgentState] = {}
    nodes = list(config.simulation.geography.nodes)
    names = ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Karl", "Leo", "Mia", "Nina", "Oscar"]
    
    for index in range(config.agents.count):
        agent_id = names[index % len(names)]
        if index >= len(names):
            agent_id = f"{agent_id}_{index}"
            
        start_position = nodes[index % len(nodes)]
        beliefs = config.agents.personal_beliefs.get(agent_id, "")
        starting_res = config.agents.starting_resources.get(agent_id, {})
        food_amount = starting_res.get("food", config.simulation.resources.get("food", 10))
        agents[agent_id] = AgentState(
            agent_id=agent_id,
            position=start_position,
            resources=ResourceBundle(
                food=food_amount,
            ),
            personal_beliefs=beliefs,
        )

    return WorldState(agents=agents, geography=graph)


def select_cooperators_pool(
    world: WorldState,
    resource_ratio: float,
    max_limit: int,
    rng: random.Random,
) -> set[str]:
    alive_agents = [agent for agent in world.agents.values() if agent.alive]
    if not alive_agents:
        return set()

    # Always include everyone when the pool is small enough
    if len(alive_agents) <= max_limit:
        return {agent.agent_id for agent in alive_agents}

    total_resources = sum(agent.resources.food for agent in alive_agents)
    num_selected = max(2, min(max_limit, int(total_resources * resource_ratio)))
    num_selected = min(num_selected, len(alive_agents))

    # Weighted shuffle without replacement to avoid duplicate agents in pool
    agents = list(alive_agents)
    weights = [max(0.001, a.resources.food) for a in agents]
    selected: list[str] = []
    while len(selected) < num_selected and agents:
        total_w = sum(weights)
        r = rng.uniform(0, total_w)
        cumulative = 0.0
        for i, (agent, w) in enumerate(zip(agents, weights)):
            cumulative += w
            if r <= cumulative:
                selected.append(agent.agent_id)
                agents.pop(i)
                weights.pop(i)
                break
    return set(selected)


def build_group_assignments(world: WorldState, config: SimulationConfig, rng: random.Random) -> list[GroupAssignment]:
    """Every alive agent is assigned exactly one group per round.
    Their partner is drawn from the other alive agents with probability
    proportional to that agent's current food resources.
    With N alive agents this always produces N group assignments.
    """
    alive_agents = [agent for agent in world.agents.values() if agent.alive]
    if len(alive_agents) < 2:
        return []

    active_ids = [a.agent_id for a in alive_agents]
    assignments: list[GroupAssignment] = []

    for group_index, agent in enumerate(alive_agents):
        # Candidates: every other alive agent
        others = [a for a in alive_agents if a.agent_id != agent.agent_id]
        weights = [max(0.001, a.resources.food) for a in others]

        # Pick one partner proportional to their food resources
        partner = rng.choices(others, weights=weights, k=1)[0]

        members = [agent.agent_id, partner.agent_id]
        multiplier = rng.uniform(
            config.simulation.grouping.public_goods_multiplier_min,
            config.simulation.grouping.public_goods_multiplier_max,
        )
        observed_by = [
            aid for aid in active_ids if aid not in members
        ]
        assignments.append(
            GroupAssignment(
                group_id=f"group_{world.step}_{group_index}",
                members=members,
                multiplier=multiplier,
                observed_by=observed_by,
            )
        )

    return assignments


def decision_context_for_agent(world: WorldState, agent_id: str, config: SimulationConfig | None = None) -> DecisionContext:
    agent = world.agents[agent_id]
    neighbors = list(world.geography.neighbors(agent.position))
    nearby_agents = [other_id for other_id, other in world.agents.items() if other_id != agent_id and other.position == agent.position and other.alive]
    visible_groups = [group for group in world.groups if agent_id in group.members or agent_id in group.observed_by]
    memory_logs = [f"Step {event.step}: {event.summary}" for event in agent.memory]
    
    depletion = config.simulation.depletion_per_round if config is not None else 1
    max_messages = config.simulation.max_messages if config is not None else 10
    other_agents_locations = {
        other_id: other.position
        for other_id, other in world.agents.items()
        if other_id != agent_id and other.alive
    }
    other_agents_resources = {
        other_id: other.resources.food
        for other_id, other in world.agents.items()
        if other_id != agent_id and other.alive
    }
    
    return DecisionContext(
        step=world.step,
        agent_id=agent_id,
        location=agent.position,
        nearby_agents=nearby_agents,
        reachable_locations=neighbors,
        visible_groups=visible_groups,
        world_summary=summarize_visible_groups(visible_groups),
        resources=agent.resources.as_dict(),
        agent_memory=memory_logs,
        personal_beliefs=agent.personal_beliefs,
        depletion_per_round=depletion,
        max_messages=max_messages,
        other_agents_locations=other_agents_locations,
        other_agents_resources=other_agents_resources,
        player_beliefs=agent.player_beliefs,
    )


def apply_group_decision(
    world: WorldState,
    config: SimulationConfig,
    rng: random.Random,
    provider: DecisionProvider,
    group: GroupAssignment,
) -> tuple[list[str], list[str]]:
    public_events: list[str] = []
    group_reports: list[str] = []

    accepted_members: list[str] = []
    contributions: dict[str, int] = {}

    import math

    for member_id in group.members:
        context = decision_context_for_agent(world, member_id, config=config)
        tools = DecisionTools(agent_id=member_id, context=context)
        decision = provider.decide(tools)
        if decision.kind == ActionKind.refuse_assignment or not decision.accepted:
            member = world.agents[member_id]
            penalty = int(math.ceil(config.simulation.rewards.refusal_penalty))
            member.resources.food = max(0, member.resources.food - penalty)
            refuse_msg = f"{member_id} refused assignment {group.group_id}"
            group_reports.append(refuse_msg)
            from .llm import log_raw_text
            log_raw_text(f"[Event] {refuse_msg}")
            member.record(
                AgentMemoryEvent(
                    step=world.step,
                    summary=f"You refused assignment {group.group_id}",
                    public=True,
                ),
                config.agents.memory_size,
            )
            continue

        accepted_members.append(member_id)
        contribution = max(0, int(decision.contribution))
        member = world.agents[member_id]
        contribution = min(contribution, member.resources.food)
        contributions[member_id] = contribution
        member.resources.food = max(0, member.resources.food - contribution)
        reasoning_str = f" (Reasoning: {decision.note})" if decision.note else ""
        member.record(
            AgentMemoryEvent(
                step=world.step,
                summary=f"Participated in {group.group_id} with contribution {contribution}{reasoning_str}",
                public=True,
            ),
            config.agents.memory_size,
        )

    if not accepted_members:
        dissolve_msg = f"{group.group_id} dissolved because nobody accepted the assignment"
        public_events.append(dissolve_msg)
        from .llm import log_raw_text
        log_raw_text(f"[Event] {dissolve_msg}")
        return public_events, group_reports

    total_contribution = sum(contributions.values())
    # Cap multiplier by number of actual cooperators
    effective_multiplier = min(group.multiplier, len(accepted_members))
    pool_return = total_contribution * effective_multiplier
    per_member_reward = int(math.floor(pool_return / len(accepted_members)))

    for member_id in accepted_members:
        member = world.agents[member_id]
        member.resources.food += per_member_reward

    result = (
        f"{group.group_id} resolved with multiplier={group.multiplier:.2f} (effective={effective_multiplier:.2f}), "
        f"contributors={contributions}, return={int(math.floor(pool_return))} (rounded down)"
    )
    public_events.append(result)
    from .llm import log_raw_text
    log_raw_text(f"[Event] {result}")
    
    # Record for observers
    for observer_id in group.observed_by:
        observer = world.agents.get(observer_id)
        if observer is not None:
            observer.record(
                AgentMemoryEvent(step=world.step, summary=result, public=True),
                config.agents.memory_size,
            )
            
    # Record for participants
    for member_id in group.members:
        member = world.agents.get(member_id)
        if member is not None and member.alive:
            member.record(
                AgentMemoryEvent(step=world.step, summary=result, public=True),
                config.agents.memory_size,
            )

    return public_events, group_reports


def handle_free_partner_selection(
    world: WorldState,
    config: SimulationConfig,
    provider: DecisionProvider,
    rng: random.Random,
) -> list[str]:
    events: list[str] = []
    candidate_pool = select_cooperators_pool(
        world,
        config.simulation.partner_selection.resource_ratio,
        config.simulation.partner_selection.max_limit,
        rng,
    )
    if len(candidate_pool) < 2:
        return events

    decisions: dict[str, AgentDecision] = {}
    for agent_id in candidate_pool:
        agent = world.agents[agent_id]
        if not agent.alive:
            continue
        context = decision_context_for_agent(world, agent_id, config=config)
        context.nearby_agents = [other_id for other_id in context.nearby_agents if other_id in candidate_pool]
        if not context.nearby_agents:
            continue
        decision = provider.decide(DecisionTools(agent_id=agent_id, context=context))
        decisions[agent_id] = decision

    cooperated_pairs = set()
    group_index = 0
    active_agents = [agent.agent_id for agent in world.agents.values() if agent.alive]

    for agent_id, decision in decisions.items():
        if decision.kind == ActionKind.select_partners:
            target_id = decision.target
            if target_id and target_id in decisions:
                target_decision = decisions[target_id]
                if (
                    target_decision.kind == ActionKind.select_partners
                    and target_decision.target == agent_id
                ):
                    pair = tuple(sorted([agent_id, target_id]))
                    if pair not in cooperated_pairs:
                        cooperated_pairs.add(pair)
                        
                        multiplier = rng.uniform(
                            config.simulation.grouping.public_goods_multiplier_min,
                            config.simulation.grouping.public_goods_multiplier_max,
                        )
                        observed_by = [
                            obs_id
                            for obs_id in active_agents
                            if obs_id not in pair
                        ]
                        
                        partner_group = GroupAssignment(
                            group_id=f"partner_group_{world.step}_{group_index}",
                            members=list(pair),
                            multiplier=multiplier,
                            observed_by=observed_by,
                        )
                        group_index += 1
                        
                        from .llm import log_raw_text
                        log_raw_text(f"[Event] {agent_id} and {target_id} mutually selected each other, forming {partner_group.group_id} (multiplier={partner_group.multiplier:.2f})")
                        
                        # Add to visible groups so the context builder can fetch it
                        world.groups.append(partner_group)
                        
                        partner_public_events, partner_reports = apply_group_decision(
                            world, config, rng, provider, partner_group
                        )
                        events.extend(partner_public_events + partner_reports)
    return events


def handle_movement(world: WorldState, config: SimulationConfig, provider: DecisionProvider, rng: random.Random) -> list[str]:
    events: list[str] = []
    for agent in world.agents.values():
        if not agent.alive:
            continue
        neighbors = list(world.geography.neighbors(agent.position))
        if not neighbors:
            continue
        context = decision_context_for_agent(world, agent.agent_id, config=config)
        decision = provider.decide(DecisionTools(agent_id=agent.agent_id, context=context))
        if decision.kind != ActionKind.move or decision.target not in neighbors:
            continue
        old_position = agent.position
        agent.position = decision.target
        move_msg = f"{agent.agent_id} moved {old_position} -> {agent.position}"
        events.append(move_msg)
        from .llm import log_raw_text
        log_raw_text(f"[Event] {move_msg}")
    return events


def handle_deliberate_conflicts(
    world: WorldState,
    config: SimulationConfig,
    provider: DecisionProvider,
    rng: random.Random,
) -> list[str]:
    events: list[str] = []
    active_agents = [agent.agent_id for agent in world.agents.values() if agent.alive]
    rng.shuffle(active_agents)

    for agent_id in active_agents:
        agent = world.agents[agent_id]
        if not agent.alive:
            continue

        context = decision_context_for_agent(world, agent_id, config=config)
        if not context.nearby_agents:
            continue

        decision = provider.decide(DecisionTools(agent_id=agent_id, context=context))
        if decision.kind == ActionKind.attack:
            target_id = decision.target
            if target_id and target_id in context.nearby_agents:
                target = world.agents[target_id]
                if not target.alive:
                    continue

                r = rng.random()
                p_attacker_kills = config.simulation.conflict.attacker_kills_probability
                p_defender_kills = config.simulation.conflict.defender_kills_probability

                if r < p_attacker_kills:
                    target.alive = False
                    loot_events = []
                    for res in ["food"]:
                        val = getattr(target.resources, res)
                        if val > 0:
                            setattr(agent.resources, res, getattr(agent.resources, res) + val)
                            setattr(target.resources, res, 0)
                            loot_events.append(f"{val} {res}")
                    loot_str = f" looting {', '.join(loot_events)}" if loot_events else ""
                    summary_msg = f"{agent_id} deliberately attacked and killed {target_id} at {agent.position}{loot_str}"
                    events.append(summary_msg)
                    from .llm import log_raw_text
                    log_msg = f"[Event] {summary_msg}"
                    if decision.note:
                        log_msg += f" (Note: {decision.note})"
                    log_raw_text(log_msg)
                    reasoning_str = f" (Reasoning: {decision.note})" if decision.note else ""
                    agent.record(
                        AgentMemoryEvent(step=world.step, summary=f"You deliberately attacked and killed {target_id} at {agent.position}{loot_str}{reasoning_str}", public=True),
                        config.agents.memory_size
                    )
                    # Notify observers
                    for other_id, other in world.agents.items():
                        if other_id not in (agent_id, target_id) and other.alive:
                            other.record(
                                AgentMemoryEvent(step=world.step, summary=summary_msg, public=True),
                                config.agents.memory_size
                            )
                elif r < p_attacker_kills + p_defender_kills:
                    agent.alive = False
                    loot_events = []
                    for res in ["food"]:
                        val = getattr(agent.resources, res)
                        if val > 0:
                            setattr(target.resources, res, getattr(target.resources, res) + val)
                            setattr(agent.resources, res, 0)
                            loot_events.append(f"{val} {res}")
                    loot_str = f" looting {', '.join(loot_events)}" if loot_events else ""
                    summary_msg = f"{agent_id} deliberately attacked {target_id} at {agent.position}, but {target_id} defended and killed {agent_id}{loot_str}"
                    events.append(summary_msg)
                    from .llm import log_raw_text
                    log_msg = f"[Event] {summary_msg}"
                    if decision.note:
                        log_msg += f" (Note: {decision.note})"
                    log_raw_text(log_msg)
                    target.record(
                        AgentMemoryEvent(step=world.step, summary=f"{agent_id} deliberately attacked you at {agent.position}, but you defended and killed them{loot_str}", public=True),
                        config.agents.memory_size
                    )
                    # Notify observers
                    for other_id, other in world.agents.items():
                        if other_id not in (agent_id, target_id) and other.alive:
                            other.record(
                                AgentMemoryEvent(step=world.step, summary=summary_msg, public=True),
                                config.agents.memory_size
                            )
                else:
                    summary_msg = f"{agent_id} deliberately attacked {target_id} at {agent.position}, but the conflict ended in a draw"
                    events.append(summary_msg)
                    from .llm import log_raw_text
                    log_msg = f"[Event] {summary_msg}"
                    if decision.note:
                        log_msg += f" (Note: {decision.note})"
                    log_raw_text(log_msg)
                    reasoning_str = f" (Reasoning: {decision.note})" if decision.note else ""
                    agent.record(
                        AgentMemoryEvent(step=world.step, summary=f"You deliberately attacked {target_id} at {agent.position}, but the conflict ended in a draw{reasoning_str}", public=True),
                        config.agents.memory_size
                    )
                    target.record(
                        AgentMemoryEvent(step=world.step, summary=f"{agent_id} deliberately attacked you at {agent.position}, but the conflict ended in a draw", public=True),
                        config.agents.memory_size
                    )
                    # Notify observers
                    for other_id, other in world.agents.items():
                        if other_id not in (agent_id, target_id) and other.alive:
                            other.record(
                                AgentMemoryEvent(step=world.step, summary=summary_msg, public=True),
                                config.agents.memory_size
                            )
    return events


def handle_communication_phase(
    world: WorldState,
    config: SimulationConfig,
    provider: DecisionProvider,
    rng: random.Random,
) -> None:
    active_agents = [agent.agent_id for agent in world.agents.values() if agent.alive]
    if len(active_agents) < 2:
        return

    message_counts = {agent_id: 0 for agent_id in active_agents}
    
    from .llm import log_raw_text
    log_raw_text(f"--- COMMUNICATION PHASE: STEP {world.step} ---")

    max_messages = config.simulation.max_messages if config is not None else 10
    for round_idx in range(max_messages):
        rng.shuffle(active_agents)
        any_messages_sent = False

        for agent_id in active_agents:
            agent = world.agents[agent_id]
            if not agent.alive:
                continue

            if message_counts[agent_id] >= max_messages:
                continue

            context = decision_context_for_agent(world, agent_id, config=config)
            tools = DecisionTools(agent_id=agent_id, context=context)
            decision = provider.decide_message(tools)

            if decision.kind == ActionKind.send_message:
                target_id = decision.target
                content = decision.content.strip()

                if len(content) > 200:
                    content = content[:200]

                if target_id and target_id in context.nearby_agents:
                    target = world.agents[target_id]
                    if target.alive:
                        any_messages_sent = True
                        message_counts[agent_id] += 1

                        log_msg = f"[Message] {agent_id} -> {target_id}: \"{content}\""
                        if decision.note:
                            log_msg += f" (Note: {decision.note})"
                        log_raw_text(log_msg)

                        target.record(
                            AgentMemoryEvent(
                                step=world.step,
                                summary=f"{agent_id} sent you a message: \"{content}\"",
                                public=False,
                            ),
                            config.agents.memory_size,
                        )

                        agent.record(
                            AgentMemoryEvent(
                                step=world.step,
                                summary=f"You sent a message to {target_id}: \"{content}\"",
                                public=False,
                            ),
                            config.agents.memory_size,
                        )
        
        if not any_messages_sent:
            break

    log_raw_text("")


def advance_world(world: WorldState, config: SimulationConfig, provider: DecisionProvider, rng: random.Random) -> WorldState:
    from .llm import log_raw_text, flush_agent_decisions
    log_raw_text("=" * 80)
    log_raw_text(f"ROUND START: STEP {world.step}")
    log_raw_text("=" * 80)
    log_raw_text("--- INITIAL AGENT STATES (RESOURCES & LOCATIONS) ---")
    for agent_id, agent in world.agents.items():
        status = "ALIVE" if agent.alive else "DEAD"
        res = agent.resources
        log_raw_text(f"Agent: {agent_id:<8} | Location: {agent.position:<3} | Status: {status:<5} | Resources: food={res.food}")
    log_raw_text("")

    # 1. Print agent memories
    log_raw_text("--- AGENT MEMORIES ---")
    for agent_id, agent in world.agents.items():
        if agent.alive:
            log_raw_text(f"Agent {agent_id} memories:")
            if agent.memory:
                for event in agent.memory:
                    log_raw_text(f"  - Step {event.step}: {event.summary}")
            else:
                log_raw_text("  - No memories yet.")
    log_raw_text("")

    # 2. Build and log group assignments before communication
    world.groups = build_group_assignments(world, config, rng)
    if world.groups:
        log_raw_text("--- GROUP ASSIGNMENTS ---")
        for group in world.groups:
            log_raw_text(f"Group {group.group_id} assigned: members={group.members}, multiplier={group.multiplier:.2f}")
        log_raw_text("")

    # 3. Communication phase
    handle_communication_phase(world, config, provider, rng)

    public_events: list[str] = []
    group_reports: list[str] = []

    for group in world.groups:
        group_public_events, group_private_reports = apply_group_decision(world, config, rng, provider, group)
        public_events.extend(group_public_events)
        group_reports.extend(group_private_reports)

    public_events.extend(handle_free_partner_selection(world, config, provider, rng))
    public_events.extend(handle_movement(world, config, provider, rng))
    conflicts = handle_deliberate_conflicts(world, config, provider, rng)

    # Apply resource depletion
    depletion = config.simulation.depletion_per_round
    if depletion > 0:
        for agent_id, agent in world.agents.items():
            if agent.alive:
                agent.resources.food = max(0, agent.resources.food - depletion)

    # Check for starvation
    for agent_id, agent in world.agents.items():
        if agent.alive and agent.resources.food <= 0:
            agent.alive = False
            starve_msg = f"{agent_id} died of starvation"
            public_events.append(starve_msg)
            log_raw_text(f"[Event] {starve_msg}.")
            for other_id, other in world.agents.items():
                if other_id != agent_id and other.alive:
                    other.record(
                        AgentMemoryEvent(
                            step=world.step,
                            summary=f"{agent_id} died of starvation at {agent.position}",
                            public=True,
                        ),
                        config.agents.memory_size,
                    )

    report = StepReport(step=world.step, public_events=public_events + group_reports, conflicts=conflicts)
    world.reports.append(report)
    world.step += 1

    # Round income is 0 food.

    log_raw_text("=" * 80)
    log_raw_text(f"ROUND END: STEP {world.step - 1}")
    log_raw_text("=" * 80)
    log_raw_text("--- POST-DEPLETION AGENT STATES ---")
    log_raw_text(f"Resource depletion: Each agent lost {depletion} units of food this round.")
    for agent_id, agent in world.agents.items():
        status = "ALIVE" if agent.alive else "DEAD"
        res = agent.resources
        log_raw_text(f"Agent: {agent_id:<8} | Location: {agent.position:<3} | Status: {status:<5} | Resources: food={res.food}")
    log_raw_text("\n")

    # Ask each alive agent to evaluate the round and update beliefs
    alive_agents = [agent_id for agent_id, agent in world.agents.items() if agent.alive]
    if alive_agents:
        log_raw_text("=" * 80)
        log_raw_text(f"AGENT ROUND EVALUATIONS & PLAN REFLECTIONS (STEP {world.step - 1})")
        log_raw_text("=" * 80)
        for agent_id in alive_agents:
            agent = world.agents[agent_id]
            context = decision_context_for_agent(world, agent_id, config=config)
            other_agents = [other_id for other_id in alive_agents if other_id != agent_id]
            
            evaluation, updated_beliefs = provider.decide_evaluation(agent_id, context, other_agents)
            
            log_raw_text(f"Agent {agent_id} round evaluation:")
            log_raw_text(f"  Evaluation: \"{evaluation}\"")
            log_raw_text("  Updated player beliefs:")
            for target_id, belief in updated_beliefs.items():
                log_raw_text(f"    - {target_id}: \"{belief}\"")
            log_raw_text("")
            
            # Persist updated beliefs in agent state
            agent.player_beliefs.update(updated_beliefs)
        log_raw_text("\n")

    flush_agent_decisions()

    return world

