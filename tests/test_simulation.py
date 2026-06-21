import random
from emergent_social_sim.config import SimulationConfig
from emergent_social_sim.models import ActionKind, AgentDecision, ResourceBundle
from emergent_social_sim.simulation import (
    create_world,
    select_cooperators_pool,
    build_group_assignments,
    apply_group_decision,
    handle_free_partner_selection,
    handle_deliberate_conflicts,
    advance_world,
)
from emergent_social_sim.policy import DecisionProvider, DecisionTools

import emergent_social_sim.llm
emergent_social_sim.llm.log_raw_text = lambda text: None


class MockDecisionProvider(DecisionProvider):
    def __init__(self, action_kind=ActionKind.wait, target=None, contribution=1):
        self.action_kind = action_kind
        self.target = target
        self.contribution = contribution

    def decide(self, tools: DecisionTools) -> AgentDecision:
        if self.action_kind == ActionKind.accept_assignment:
            return tools.accept_assignment(contribution=self.contribution)
        elif self.action_kind == ActionKind.select_partners:
            # target must be in tools.context.nearby_agents
            t = self.target if self.target in tools.context.nearby_agents else (tools.context.nearby_agents[0] if tools.context.nearby_agents else None)
            return tools.select_partners(target=t)
        elif self.action_kind == ActionKind.attack:
            t = self.target if self.target in tools.context.nearby_agents else (tools.context.nearby_agents[0] if tools.context.nearby_agents else None)
            return tools.attack(target=t)
        elif self.action_kind == ActionKind.move:
            t = tools.context.reachable_locations[0] if tools.context.reachable_locations else "A"
            return tools.move(target=t)
        return tools.wait()


def test_create_world_has_symmetric_resources_and_names():
    config = SimulationConfig()
    world = create_world(config)
    
    # 1. Names check
    names = list(world.agents.keys())
    assert "Alice" in names
    assert "Bob" in names
    
    # 2. Symmetric resources check
    alice = world.agents["Alice"]
    assert alice.resources.food == 10.0
    assert not hasattr(alice.resources, "water")
    assert not hasattr(alice.resources, "salt")
    assert not hasattr(alice.resources, "meat")
    assert not hasattr(alice.resources, "labor")
    assert not hasattr(alice.resources, "material")
    assert not hasattr(alice, "reputation")


def test_select_cooperators_pool():
    config = SimulationConfig()
    config.simulation.grouping.resource_ratio = 0.1
    config.simulation.grouping.max_limit = 4
    world = create_world(config)
    
    rng = random.Random(42)
    pool = select_cooperators_pool(world, config.simulation.grouping.resource_ratio, config.simulation.grouping.max_limit, rng)
    # Total starting resources per agent = 10. Total = 6 * 10 = 60.
    # 60 * 0.1 = 6 cooperators sampled, capped at max_limit = 4.
    assert len(pool) <= 4
    assert all(name in world.agents for name in pool)


def test_build_group_assignments_capped_at_2():
    config = SimulationConfig()
    config.simulation.grouping.resource_ratio = 0.5
    config.simulation.grouping.max_limit = 10
    config.simulation.grouping.assignment_probability = 1.0
    world = create_world(config)
    
    rng = random.Random(42)
    groups = build_group_assignments(world, config, rng)
    
    for group in groups:
        assert len(group.members) == 2


def test_multiplier_cap_in_group_decision():
    config = SimulationConfig()
    config.simulation.grouping.public_goods_multiplier_min = 5.0
    config.simulation.grouping.public_goods_multiplier_max = 5.0
    world = create_world(config)
    
    # Group size is 2, multiplier is 5.0, but effective multiplier must be capped at len(cooperators) = 2.
    from emergent_social_sim.models import GroupAssignment
    group = GroupAssignment(group_id="test_g", members=["Alice", "Bob"], multiplier=5.0)
    
    rng = random.Random(42)
    provider = MockDecisionProvider(action_kind=ActionKind.accept_assignment, contribution=1)
    
    # Starting food: Alice=10, Bob=10
    public_events, group_reports = apply_group_decision(world, config, rng, provider, group)
    
    # Contrib: Alice=1.0, Bob=1.0. Total = 2.0
    # Multiplier: 5.0. Effective cap: 2.0 (number of accepted members)
    # Pool return: 2.0 * 2.0 = 4.0. Reward per member: 4.0 / 2 = 2
    # Final food: Alice = 10 - 1 + 2 = 11, Bob = 10 - 1 + 2 = 11
    assert world.agents["Alice"].resources.food == 11
    assert world.agents["Bob"].resources.food == 11


def test_bilateral_partner_selection():
    config = SimulationConfig()
    config.agents.count = 2
    config.simulation.grouping.public_goods_multiplier_min = 2.0
    config.simulation.grouping.public_goods_multiplier_max = 2.0
    config.simulation.partner_selection.resource_ratio = 1.0
    config.simulation.partner_selection.max_limit = 10
    config.simulation.rewards.cooperation_bonus = 0.2
    
    world = create_world(config)
    # Place Alice and Bob in the same starting position
    world.agents["Alice"].position = "A"
    world.agents["Bob"].position = "A"
    
    # Provider: Alice chooses Bob, Bob chooses Alice. Then they accept with 1 contribution.
    class BilateralMockProvider(DecisionProvider):
        def decide(self, tools: DecisionTools) -> AgentDecision:
            if tools.context.visible_groups:
                return tools.accept_assignment(contribution=1)
            if tools.agent_id == "Alice":
                return tools.select_partners(target="Bob")
            elif tools.agent_id == "Bob":
                return tools.select_partners(target="Alice")
            return tools.wait()
            
    rng = random.Random(42)
    events = handle_free_partner_selection(world, config, BilateralMockProvider(), rng)
    # The dynamically created partner group resolves
    assert "partner_group_0_0 resolved with multiplier=2.00 (effective=2.00)" in events[0]
    # Alice food: 10 - 1 + 2 = 11. Bob food: 10 - 1 + 2 = 11.
    assert world.agents["Alice"].resources.food == 11
    assert world.agents["Bob"].resources.food == 11



def test_deliberate_conflicts():
    config = SimulationConfig()
    config.agents.count = 2
    config.simulation.conflict.attacker_kills_probability = 1.0  # 100% attacker kills
    config.simulation.conflict.defender_kills_probability = 0.0
    
    world = create_world(config)
    world.agents["Alice"].position = "A"
    world.agents["Bob"].position = "A"
    
    # Let Alice have food=10, Bob have food=5
    world.agents["Alice"].resources.food = 10
    world.agents["Bob"].resources.food = 5
    
    # Provider: Alice attacks Bob
    class AttackProvider(DecisionProvider):
        def decide(self, tools: DecisionTools) -> AgentDecision:
            if tools.agent_id == "Alice":
                return tools.attack(target="Bob")
            return tools.wait()
            
    rng = random.Random(42)
    events = handle_deliberate_conflicts(world, config, AttackProvider(), rng)
    
    assert "Alice deliberately attacked and killed Bob at A looting" in events[0]
    assert not world.agents["Bob"].alive
    # Alice looted food
    assert world.agents["Alice"].resources.food == 15
    assert world.agents["Bob"].resources.food == 0


def test_agent_memory_recording():
    config = SimulationConfig()
    config.agents.count = 3
    world = create_world(config)
    
    world.agents["Alice"].position = "A"
    world.agents["Bob"].position = "A"
    world.agents["Charlie"].position = "A"
    
    from emergent_social_sim.models import GroupAssignment
    group = GroupAssignment(group_id="test_group_mem", members=["Alice", "Bob"], multiplier=2.0)
    
    provider = MockDecisionProvider(action_kind=ActionKind.accept_assignment, contribution=1)
    rng = random.Random(42)
    apply_group_decision(world, config, rng, provider, group)
    
    alice_memories = [m.summary for m in world.agents["Alice"].memory]
    bob_memories = [m.summary for m in world.agents["Bob"].memory]
    
    assert any("test_group_mem resolved" in m for m in alice_memories)
    assert any("test_group_mem resolved" in m for m in bob_memories)

    config.simulation.conflict.attacker_kills_probability = 1.0
    config.simulation.conflict.defender_kills_probability = 0.0
    
    class AttackProvider(DecisionProvider):
        def decide(self, tools: DecisionTools) -> AgentDecision:
            if tools.agent_id == "Alice":
                return tools.attack(target="Bob")
            return tools.wait()
            
    handle_deliberate_conflicts(world, config, AttackProvider(), rng)
    
    alice_memories_post = [m.summary for m in world.agents["Alice"].memory]
    charlie_memories = [m.summary for m in world.agents["Charlie"].memory]
    
    assert any("You deliberately attacked and killed Bob" in m for m in alice_memories_post)
    assert any("Alice deliberately attacked and killed Bob" in m for m in charlie_memories)


def test_communication_phase():
    config = SimulationConfig()
    config.agents.count = 2
    world = create_world(config)
    world.agents["Alice"].position = "A"
    world.agents["Bob"].position = "A"

    class CommunicationMockProvider(DecisionProvider):
        def decide(self, tools: DecisionTools) -> AgentDecision:
            return tools.wait()
        
        def decide_message(self, tools: DecisionTools) -> AgentDecision:
            if tools.agent_id == "Alice":
                return tools.send_message(target="Bob", content="a" * 105)
            return tools.wait()

    from emergent_social_sim.simulation import handle_communication_phase
    rng = random.Random(42)
    handle_communication_phase(world, config, CommunicationMockProvider(), rng)

    bob_memories = [m.summary for m in world.agents["Bob"].memory]
    expected_bob_msg = f"Alice sent you a message: \"{'a' * 100}\""
    assert any(expected_bob_msg in m for m in bob_memories)

    alice_memories = [m.summary for m in world.agents["Alice"].memory]
    expected_alice_msg = f"You sent a message to Bob: \"{'a' * 100}\""
    assert any(expected_alice_msg in m for m in alice_memories)


def test_starvation():
    config = SimulationConfig()
    config.agents.count = 2
    world = create_world(config)
    # Put Alice and Bob at the same location A
    world.agents["Alice"].position = "A"
    world.agents["Bob"].position = "A"
    # Give Alice 0 food and Bob 5 food
    world.agents["Alice"].resources.food = 0
    world.agents["Bob"].resources.food = 5
    
    # We will advance the world. Mock decisions will wait.
    provider = MockDecisionProvider(action_kind=ActionKind.wait)
    rng = random.Random(42)
    world = advance_world(world, config, provider, rng)
    
    assert not world.agents["Alice"].alive
    assert world.agents["Bob"].alive
    # Bob should have a memory event about Alice starving
    bob_memories = [m.summary for m in world.agents["Bob"].memory]
    assert any("Alice died of starvation at A" in m for m in bob_memories)


def test_personal_beliefs():
    config = SimulationConfig()
    config.agents.personal_beliefs = {"Alice": "Alice's secret belief"}
    world = create_world(config)
    
    assert world.agents["Alice"].personal_beliefs == "Alice's secret belief"
    assert world.agents["Bob"].personal_beliefs == ""
    
    from emergent_social_sim.simulation import decision_context_for_agent
    context = decision_context_for_agent(world, "Alice")
    assert context.personal_beliefs == "Alice's secret belief"


if __name__ == "__main__":
    test_create_world_has_symmetric_resources_and_names()
    test_select_cooperators_pool()
    test_build_group_assignments_capped_at_2()
    test_multiplier_cap_in_group_decision()
    test_bilateral_partner_selection()
    test_deliberate_conflicts()
    test_agent_memory_recording()
    test_communication_phase()
    test_starvation()
    test_personal_beliefs()
    print("All tests passed successfully!")
