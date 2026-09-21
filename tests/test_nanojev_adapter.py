import unittest

from arena import ArenaConfig, ArenaEnv
from agents.nanojev_agent import NanoJevAgent
from nanojev_adapter.schema import decision_request, parse_probabilities
from nanojev_adapter.policy import greedy_avoid_backtrack, select_action


class FakeClient:
    def __init__(self):
        self.calls = 0

    def evaluate(self, payload):
        self.calls += 1
        states = []
        for state in payload["states"]:
            candidates = state["questions"]["action"]["criteria"]
            chosen = next(iter(candidates))
            probabilities = {key: float(key == chosen) for key in candidates}
            states.append({"id": state["id"],
                           "answers": {"action": {"type": "choice", "probabilities": probabilities}}})
        return {"states": states,
                "execution": {"forward_passes": 1, "network_model_calls": 0}}


class NanoJevAdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=1, fires=1, medkits=0))

    def test_request_matches_upstream_contract(self):
        payload = decision_request(self.env)
        state = payload["states"][0]
        self.assertEqual(set(state), {"id", "state", "questions"})
        self.assertEqual(state["questions"]["action"]["type"], "choice")

    def test_probability_validation_rejects_missing_candidate(self):
        response = {"states": [{"answers": {"action": {"probabilities": {"wait": 1.0}}}}]}
        with self.assertRaises(ValueError):
            parse_probabilities(response, {"wait", "move_n"})

    def test_agent_uses_real_distribution_without_fallback(self):
        agent = NanoJevAgent(FakeClient())
        action = agent.act(self.env)
        self.assertIn(action, self.env.legal_actions())
        self.assertAlmostEqual(sum(agent.last_probabilities.values()), 1)

    def test_forced_action_does_not_call_model(self):
        env = ArenaEnv(ArenaConfig(width=1, height=1, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        client = FakeClient()
        agent = NanoJevAgent(client, max_batch_states=2)
        self.assertEqual(agent.act(env).value, "wait")
        self.assertEqual(client.calls, 0)
        self.assertEqual(agent.last_probabilities, {"wait": 1.0})

    def test_policy_avoids_immediate_backtrack(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.step("move_e")
        action, reason = greedy_avoid_backtrack({"move_w": .7, "move_n": .2, "wait": .1}, env)
        self.assertEqual((action, reason), ("move_n", "backtrack_avoided"))

    def test_policy_uses_model_probability_plus_gem_progress(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.gems = {(4, 2)}
        action, reason = select_action({"move_w": .4, "move_e": .3, "wait": .3}, env, "hybrid")
        self.assertEqual((action, reason), ("move_e", "planner_rerank"))

    def test_hybrid_routes_around_wall_instead_of_following_probability(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.gems, env.walls = {(4, 2)}, {(3, 2)}
        action, reason = select_action({"move_w": .98, "move_n": .015, "move_s": .005}, env, "hybrid")
        self.assertEqual((action, reason), ("move_n", "planner_route"))

    def test_hybrid_routes_to_missing_weapon_before_gem(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.gems, env.bow_pickups = {(0, 2)}, {(4, 2)}
        action, reason = select_action({"move_w": .9, "move_e": .1}, env, "hybrid")
        self.assertEqual((action, reason), ("move_e", "planner_route"))

    def test_hybrid_uses_aligned_ranged_weapon(self):
        env = ArenaEnv(ArenaConfig(width=8, height=3, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        action, reason = select_action({"move_s": .18, "shoot_bow_e": .11,
                                        "shoot_pistol_e": .08}, env, "hybrid")
        self.assertEqual((action, reason), ("shoot_bow_e", "planner_rerank"))

    def test_hybrid_immediately_heals_at_critical_health(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.hp, env.player.medkits = 30, 1
        action, reason = select_action({"move_e": .9, "heal": .1}, env, "hybrid")
        self.assertEqual((action, reason), ("heal", "survival_heal"))

    def test_hybrid_routes_to_ground_medkit_before_gem_when_low(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position, env.player.hp = (2, 2), 40
        env.gems, env.medkits = {(0, 2)}, {(4, 2)}
        action, reason = select_action({"move_w": .9, "move_e": .1}, env, "hybrid")
        self.assertEqual((action, reason), ("move_e", "planner_route"))

    def test_model_policy_preserves_argmax(self):
        action, reason = select_action({"move_w": .6, "move_e": .4}, self.env, "model")
        self.assertEqual((action, reason), ("move_w", "model_argmax"))

    def test_hybrid_does_not_wait_twice_when_safe_move_exists(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.step("wait")
        action, reason = select_action({"wait": .9, "move_n": .1}, env, "hybrid")
        self.assertEqual((action, reason), ("move_n", "planner_rerank"))

    def test_hybrid_does_not_override_dash_with_gem_route(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.gems = {(4, 2)}
        action, reason = select_action({"dash_e": .9, "move_e": .1}, env, "hybrid")
        self.assertEqual((action, reason), ("dash_e", "model_argmax"))

    def test_agent_batches_multiple_states_in_one_call(self):
        client = FakeClient()
        agent = NanoJevAgent(client, max_batch_states=2)
        actions = agent.act_many([self.env, self.env.clone()])
        self.assertEqual(len(actions), 2)
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(agent.last_batch_probabilities), 2)
        self.assertEqual(len(agent.last_batch_state_latency_ms), 2)

    def test_agent_chunks_batches_for_small_gpu(self):
        client = FakeClient()
        agent = NanoJevAgent(client, max_batch_states=2)
        agent.act_many([self.env, self.env.clone(), self.env.clone()])
        self.assertEqual(client.calls, 2)

    def test_agent_caches_identical_decision_request(self):
        client = FakeClient()
        agent = NanoJevAgent(client)
        agent.act(self.env)
        agent.act(self.env)
        self.assertEqual(client.calls, 1)
        self.assertEqual(agent.last_cache_hits, 1)
        self.assertEqual(agent.last_latency_ms, 0)


if __name__ == "__main__":
    unittest.main()
