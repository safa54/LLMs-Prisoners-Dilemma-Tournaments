from __future__ import annotations

import unittest
from simple_pd.config import PayoutMatrix, GameConfig, TournamentConfig, AgentConfig
from simple_pd.agent import AlwaysCooperateBot, AlwaysDefectBot, TitForTatBot
from simple_pd.game import PDGame
from simple_pd.tournament import PDTournament, build_agent


class TestSimplePD(unittest.TestCase):
    def test_payout_validation(self):
        # Valid classic PD payoffs
        payouts = PayoutMatrix(T=5, R=3, P=1, S=0)
        self.assertEqual(payouts.T, 5)

        # Invalid: T not greater than R
        with self.assertRaises(ValueError):
            PayoutMatrix(T=3, R=3, P=1, S=0)

        # Invalid: 2*R <= T + S (not classic PD)
        with self.assertRaises(ValueError):
            PayoutMatrix(T=6, R=3, P=1, S=0)

    def test_deterministic_bots(self):
        payouts = PayoutMatrix()
        history = []
        
        coop_bot = AlwaysCooperateBot("Coop")
        defect_bot = AlwaysDefectBot("Defect")
        tft_bot = TitForTatBot("TFT")

        # Round 1
        c_coop, _ = coop_bot.make_decision(history, "other", payouts)
        c_defect, _ = defect_bot.make_decision(history, "other", payouts)
        c_tft, _ = tft_bot.make_decision(history, "other", payouts)

        self.assertEqual(c_coop, "cooperate")
        self.assertEqual(c_defect, "defect")
        self.assertEqual(c_tft, "cooperate")

        # Round 2 after opponent defected
        history_tft = [{"my_choice": "cooperate", "opponent_choice": "defect", "my_payoff": 0, "opponent_payoff": 5}]
        c_tft_2, _ = tft_bot.make_decision(history_tft, "other", payouts)
        self.assertEqual(c_tft_2, "defect")

        # Round 3 after opponent cooperated
        history_tft.append({"my_choice": "defect", "opponent_choice": "cooperate", "my_payoff": 5, "opponent_payoff": 0})
        c_tft_3, _ = tft_bot.make_decision(history_tft, "other", payouts)
        self.assertEqual(c_tft_3, "cooperate")

    def test_pd_game(self):
        config = GameConfig(rounds=5)
        
        # Game 1: Coop vs Defect
        agent1 = AlwaysCooperateBot("Coop")
        agent2 = AlwaysDefectBot("Defect")
        game = PDGame(agent1, agent2, config)
        results = game.run()

        self.assertEqual(results["agent1_score"], 0)
        self.assertEqual(results["agent2_score"], 25)
        self.assertEqual(results["agent1_cooperation_rate"], 1.0)
        self.assertEqual(results["agent2_cooperation_rate"], 0.0)

        # Game 2: TFT vs TFT
        tft1 = TitForTatBot("TFT1")
        tft2 = TitForTatBot("TFT2")
        game2 = PDGame(tft1, tft2, config)
        results2 = game2.run()

        self.assertEqual(results2["agent1_score"], 15)
        self.assertEqual(results2["agent2_score"], 15)

    def test_pd_tournament(self):
        config = TournamentConfig(
            game=GameConfig(rounds=5),
            agents=[
                AgentConfig(agent_id="AlwaysCoop", agent_type="bot", bot_type="always_cooperate"),
                AgentConfig(agent_id="AlwaysDefect", agent_type="bot", bot_type="always_defect"),
                AgentConfig(agent_id="TitForTat", agent_type="bot", bot_type="tit_for_tat"),
            ],
            log_file=None
        )

        tournament = PDTournament(config)
        summary = tournament.run()
        standings = summary["standings"]

        # Check total scores mathematically:
        # Coop vs Defect: Coop=0, Defect=25
        # Coop vs TFT: Coop=15, TFT=15
        # Defect vs TFT: Defect=9, TFT=4
        # Total scores:
        # AlwaysDefect = 25 + 9 = 34
        # TitForTat = 15 + 4 = 19
        # AlwaysCoop = 0 + 15 = 15

        self.assertEqual(standings[0]["agent_id"], "AlwaysDefect")
        self.assertEqual(standings[0]["total_score"], 34)
        self.assertEqual(standings[0]["cooperation_rate"], 0.0)

        self.assertEqual(standings[1]["agent_id"], "TitForTat")
        self.assertEqual(standings[1]["total_score"], 19)
        self.assertEqual(standings[1]["cooperation_rate"], 0.6)

        self.assertEqual(standings[2]["agent_id"], "AlwaysCoop")
        self.assertEqual(standings[2]["total_score"], 15)
        self.assertEqual(standings[2]["cooperation_rate"], 1.0)

    def test_new_bots(self):
        payouts = PayoutMatrix()
        
        from simple_pd.agent import ForgivingTitForTatBot, ManipulativeTitForTatBot, RandomBot, TitForTwoTatsBot
        
        # 1. Test Forgiving TFT (forgives on every 3rd opponent defection)
        forgiving = ForgivingTitForTatBot("Forgiving")
        self.assertEqual(forgiving.make_decision([], "other", payouts)[0], "cooperate")
        
        hist = [{"my_choice": "cooperate", "opponent_choice": "defect", "my_payoff": 0, "opponent_payoff": 5}]
        self.assertEqual(forgiving.make_decision(hist, "other", payouts)[0], "defect")
        
        hist.append({"my_choice": "defect", "opponent_choice": "defect", "my_payoff": 1, "opponent_payoff": 1})
        self.assertEqual(forgiving.make_decision(hist, "other", payouts)[0], "defect")
        
        hist.append({"my_choice": "defect", "opponent_choice": "defect", "my_payoff": 1, "opponent_payoff": 1})
        self.assertEqual(forgiving.make_decision(hist, "other", payouts)[0], "cooperate")
        
        # 2. Test Manipulative TFT (manipulates/defects on every 3rd opponent cooperation)
        manipulator = ManipulativeTitForTatBot("Manipulator")
        self.assertEqual(manipulator.make_decision([], "other", payouts)[0], "cooperate")
        
        hist_m = [{"my_choice": "cooperate", "opponent_choice": "cooperate", "my_payoff": 3, "opponent_payoff": 3}]
        self.assertEqual(manipulator.make_decision(hist_m, "other", payouts)[0], "cooperate")
        
        hist_m.append({"my_choice": "cooperate", "opponent_choice": "cooperate", "my_payoff": 3, "opponent_payoff": 3})
        self.assertEqual(manipulator.make_decision(hist_m, "other", payouts)[0], "cooperate")
        
        hist_m.append({"my_choice": "cooperate", "opponent_choice": "cooperate", "my_payoff": 3, "opponent_payoff": 3})
        self.assertEqual(manipulator.make_decision(hist_m, "other", payouts)[0], "defect")
        
        # 3. Test TitForTwoTatsBot (only defects if opponent defects twice in a row)
        tf2t = TitForTwoTatsBot("TF2T")
        self.assertEqual(tf2t.make_decision([], "other", payouts)[0], "cooperate")
        
        hist_2t = [{"my_choice": "cooperate", "opponent_choice": "defect", "my_payoff": 0, "opponent_payoff": 5}]
        self.assertEqual(tf2t.make_decision(hist_2t, "other", payouts)[0], "cooperate")
        
        hist_2t.append({"my_choice": "cooperate", "opponent_choice": "defect", "my_payoff": 0, "opponent_payoff": 5})
        self.assertEqual(tf2t.make_decision(hist_2t, "other", payouts)[0], "defect")
        
        # 4. Test RandomBot
        random_bot = RandomBot("Random")
        choice, _ = random_bot.make_decision([], "other", payouts)
        self.assertIn(choice, ["cooperate", "defect"])

    def test_match_cache(self):
        import tempfile
        import shutil
        from pathlib import Path
        from simple_pd.tournament import MatchCache
        
        # Create a temporary directory for cache file
        temp_dir = tempfile.mkdtemp()
        try:
            cache_file = Path(temp_dir) / "test_cache.json"
            cache = MatchCache(str(cache_file))
            
            game_config = GameConfig(rounds=2)
            
            # 1. Test key and swap logic
            key_normal, swapped_normal = cache._get_key_and_swapped("AgentA", "AgentB", game_config)
            key_rev, swapped_rev = cache._get_key_and_swapped("AgentB", "AgentA", game_config)
            
            self.assertEqual(key_normal, key_rev)
            self.assertFalse(swapped_normal)
            self.assertTrue(swapped_rev)
            
            # 2. Add some game result
            dummy_result = {
                "agent1_id": "AgentA",
                "agent2_id": "AgentB",
                "agent1_score": 5,
                "agent2_score": 3,
                "agent1_cooperation_rate": 0.5,
                "agent2_cooperation_rate": 1.0,
                "rounds_played": 2,
                "history": [
                    {
                        "round": 1,
                        "my_choice": "defect",
                        "opponent_choice": "cooperate",
                        "my_payoff": 5,
                        "opponent_payoff": 0,
                        "my_reasoning": "ReasonA1",
                        "opponent_reasoning": "ReasonB1",
                    },
                    {
                        "round": 2,
                        "my_choice": "cooperate",
                        "opponent_choice": "cooperate",
                        "my_payoff": 3,
                        "opponent_payoff": 3,
                        "my_reasoning": "ReasonA2",
                        "opponent_reasoning": "ReasonB2",
                    }
                ]
            }
            
            cache.set("AgentA", "AgentB", game_config, dummy_result)
            self.assertTrue(cache_file.exists())
            
            # 3. Retrieve in same order
            res_same = cache.get("AgentA", "AgentB", game_config)
            self.assertEqual(res_same["agent1_id"], "AgentA")
            self.assertEqual(res_same["agent2_id"], "AgentB")
            self.assertEqual(res_same["agent1_score"], 5)
            self.assertEqual(res_same["agent2_score"], 3)
            self.assertEqual(res_same["agent1_cooperation_rate"], 0.5)
            self.assertEqual(res_same["agent2_cooperation_rate"], 1.0)
            self.assertEqual(res_same["history"][0]["my_choice"], "defect")
            self.assertEqual(res_same["history"][0]["opponent_choice"], "cooperate")
            
            # 4. Retrieve in swapped order
            res_swapped = cache.get("AgentB", "AgentA", game_config)
            self.assertEqual(res_swapped["agent1_id"], "AgentB")
            self.assertEqual(res_swapped["agent2_id"], "AgentA")
            self.assertEqual(res_swapped["agent1_score"], 3)
            self.assertEqual(res_swapped["agent2_score"], 5)
            self.assertEqual(res_swapped["agent1_cooperation_rate"], 1.0)
            self.assertEqual(res_swapped["agent2_cooperation_rate"], 0.5)
            
            # Check history swapped perspective
            self.assertEqual(res_swapped["history"][0]["my_choice"], "cooperate")
            self.assertEqual(res_swapped["history"][0]["opponent_choice"], "defect")
            self.assertEqual(res_swapped["history"][0]["my_payoff"], 0)
            self.assertEqual(res_swapped["history"][0]["opponent_payoff"], 5)
            self.assertEqual(res_swapped["history"][0]["my_reasoning"], "ReasonB1")
            self.assertEqual(res_swapped["history"][0]["opponent_reasoning"], "ReasonA1")
            
            # Check that setting in reverse order normalized it correctly
            cache2 = MatchCache(str(cache_file))
            dummy_result_rev = {
                "agent1_id": "AgentB",
                "agent2_id": "AgentA",
                "agent1_score": 3,
                "agent2_score": 5,
                "agent1_cooperation_rate": 1.0,
                "agent2_cooperation_rate": 0.5,
                "rounds_played": 2,
                "history": [
                    {
                        "round": 1,
                        "my_choice": "cooperate",
                        "opponent_choice": "defect",
                        "my_payoff": 0,
                        "opponent_payoff": 5,
                        "my_reasoning": "ReasonB1",
                        "opponent_reasoning": "ReasonA1",
                    },
                    {
                        "round": 2,
                        "my_choice": "cooperate",
                        "opponent_choice": "cooperate",
                        "my_payoff": 3,
                        "opponent_payoff": 3,
                        "my_reasoning": "ReasonB2",
                        "opponent_reasoning": "ReasonA2",
                    }
                ]
            }
            cache2.set("AgentB", "AgentA", game_config, dummy_result_rev)
            
            # Retrieve again and ensure it parses same as original
            res_same2 = cache2.get("AgentA", "AgentB", game_config)
            self.assertEqual(res_same2["agent1_score"], 5)
            self.assertEqual(res_same2["agent2_score"], 3)
            
        finally:
            shutil.rmtree(temp_dir)

    def test_cache_handles_api_failure(self):
        import tempfile
        import shutil
        from pathlib import Path
        from simple_pd.tournament import PDTournament
        
        temp_dir = tempfile.mkdtemp()
        try:
            cache_file = Path(temp_dir) / "test_api_fail_cache.json"
            
            # Setup a tournament where one agent fails API call
            config = TournamentConfig(
                game=GameConfig(rounds=2),
                agents=[
                    AgentConfig(agent_id="AlwaysCoop", agent_type="bot", bot_type="always_cooperate"),
                    # We configure an LLM agent with invalid API key to force failure
                    AgentConfig(
                        agent_id="LLMFail", 
                        agent_type="llm", 
                        model_name="gpt-4o-mini", 
                        api_key="invalid-key-to-force-failure"
                    ),
                ],
                match_cache_file=str(cache_file)
            )
            
            tournament = PDTournament(config)
            summary = tournament.run()
            
            # Since LLMFail fails, the results should NOT be cached
            if cache_file.exists():
                import json
                cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
                self.assertEqual(len(cache_data), 0)
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
