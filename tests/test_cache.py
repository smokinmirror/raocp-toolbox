import unittest
import raocp.core.scenario_tree as core_tree
import raocp.core.problem_spec as core_spec
import raocp.core.cache as core_cache
import numpy as np


class TestCache(unittest.TestCase):
    __tree_from_markov = None
    __raocp_from_markov = None
    __good_size = 3

    @staticmethod
    def _construct_tree_from_markov():
        if TestCache.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.5, 0])
            (N, tau) = (4, 3)
            TestCache.__tree_from_markov = core_tree.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

    @staticmethod
    def _construct_raocp_from_markov():
        if TestCache.__raocp_from_markov is None:
            tree = TestCache.__tree_from_markov

            # construct markovian set of system and control dynamics
            system = np.eye(TestCache.__good_size)
            set_system = [system, 2 * system, 3 * system]  # n x n matrices
            control = np.eye(TestCache.__good_size)
            set_control = [control, 2 * control, 3 * control]  # n x u matrices

            # construct cost weight matrices
            cost_type = "Quadratic"
            cost_types = [cost_type] * TestCache.__good_size
            nonleaf_state_weight = 10 * np.eye(TestCache.__good_size)  # n x n matrix
            nonleaf_state_weights = [nonleaf_state_weight, 2 * nonleaf_state_weight, 3 * nonleaf_state_weight]
            control_weight = np.eye(TestCache.__good_size)  # u x u matrix OR scalar
            control_weights = [control_weight, 2 * control_weight, 3 * control_weight]
            leaf_state_weight = 5 * np.eye(TestCache.__good_size)  # n x n matrix

            # define risks
            (risk_type, alpha) = ("AVaR", 0.5)

            TestCache.__raocp_from_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_dynamics(set_system, set_control) \
                .with_markovian_costs(cost_types, nonleaf_state_weights, control_weights) \
                .with_all_leaf_costs(cost_type, leaf_state_weight) \
                .with_all_risks(risk_type, alpha)

    @staticmethod
    def _construct_mock_cache():
        mock_cache = core_cache.Cache(TestCache.__raocp_from_markov)
        primal_segments = mock_cache.get_primal_segments()
        dual_segments = mock_cache.get_dual_segments()
        return mock_cache, primal_segments, dual_segments

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestCache._construct_tree_from_markov()
        TestCache._construct_raocp_from_markov()

    def test_proximal_of_relaxation_s_at_stage_zero(self):
        mock_cache, seg_p, seg_d = self._construct_mock_cache()
        # s = segment 5
        s = 5
        parameter = np.random.randn(mock_cache._Cache__primal[seg_p[s]].size)
        mock_cache.proximal_of_relaxation_s_at_stage_zero(parameter)
        self.assertEqual(-parameter, mock_cache._Cache__primal[seg_p[s]][0])

    def test_


if __name__ == '__main__':
    unittest.main()
