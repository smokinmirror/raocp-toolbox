import unittest
import raocp.core.scenario_tree as core_tree
import raocp.core.problem_spec as core_spec
import raocp.core.cache as core_cache
import numpy as np
import cvxpy as cp


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

    def test_project_on_dynamics(self):
        mock_cache, seg_p, _ = self._construct_mock_cache()

        # solve with cvxpy first
        _, prim = mock_cache.get_primal()  # template
        for i in range(seg_p[1], seg_p[3]):
            prim[i] = np.random.randn(prim[i].size)

        x_bar = np.asarray(prim[seg_p[1]: seg_p[2]])
        u_bar = np.asarray(prim[seg_p[2]: seg_p[3]])
        N = self.__tree_from_markov.num_nodes
        n = self.__tree_from_markov.num_nonleaf_nodes
        x = cp.Variable(x_bar.shape)
        u = cp.Variable(u_bar.shape)
        # sum problem objectives and concatenate constraints
        cost = 0
        constraints = [x[0, :] == x_bar[0, :]]
        # nonleaf nodes
        for node in range(n):
            cost += cp.sum_squares(x[node, :] - x_bar[node, :]) + cp.sum_squares(u[node, :] - u_bar[node, :])
            for ch in self.__tree_from_markov.children_of(node):
                constraints += [x[ch, :] ==
                                self.__raocp_from_markov.state_dynamics_at_node(ch) @ x[node, :] +
                                self.__raocp_from_markov.control_dynamics_at_node(ch) @ u[node, :]]

        # leaf nodes
        for node in range(n, N):
            cost += cp.sum_squares(x[node, :] - x_bar[node, :])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.ECOS)
        # ensure x0 stayed the same
        self.assertTrue(np.allclose(prim[seg_p[1]], x.value[0, :]))

        # solve with dp
        for i in range(seg_p[1], seg_p[3]):
            prim[i] = prim[i].reshape(-1, 1)

        mock_cache.cache_initial_state(prim[seg_p[1]])
        mock_cache.set_primal(prim)
        mock_cache.project_on_dynamics()
        dp_result, _ = mock_cache.get_primal()
        x_dp = np.asarray(prim[seg_p[1]: seg_p[2]])
        u_dp = np.asarray(prim[seg_p[2]: seg_p[3]])
        # ensure x0 stayed the same
        self.assertTrue(np.allclose(prim[seg_p[1]], x_dp[0, :]))

        # check solutions are similar
        node = 1
        print(f"cvxpy x = {u.value[node]}\n"
              f"dp x = {u_dp[node].T}")
        self.assertTrue(np.allclose(x.value, x_dp[:, :, 0]))
        self.assertTrue(np.allclose(u.value, u_dp[:, :, 0]))


if __name__ == '__main__':
    unittest.main()
