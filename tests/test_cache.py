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
            (N, tau) = (1, 1)
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

    def test_cache_initial_state(self):
        mock_cache, seg_p, seg_d = self._construct_mock_cache()
        _, prim = mock_cache.get_primal()  # template
        prim[seg_p[1]] = np.random.randn(prim[seg_p[1]].size).reshape(-1, 1)
        mock_cache.cache_initial_state(prim[seg_p[1]])
        _, old_prim = mock_cache.get_primal()
        self.assertTrue(np.array_equal(prim[seg_p[1]], mock_cache._Cache__initial_state))
        self.assertTrue(np.array_equal(prim[seg_p[1]], old_prim[0]))
        self.assertTrue(np.array_equal(prim[seg_p[1]], mock_cache._Cache__primal_cache[0][0]))

    def test_proximal_of_relaxation_s_at_stage_zero(self):
        mock_cache, seg_p, seg_d = self._construct_mock_cache()
        # s = segment 5
        s = 5
        _, prim = mock_cache.get_primal()  # get old primal as template
        parameter = np.random.randn(prim[seg_p[s]].size)
        mock_cache.proximal_of_relaxation_s_at_stage_zero(parameter)
        prim, _ = mock_cache.get_primal()  # get modified primal
        self.assertEqual(-parameter, prim[seg_p[s]][0])

    def test_project_on_dynamics(self):
        mock_cache, seg_p, _ = self._construct_mock_cache()
        _, prim = mock_cache.get_primal()  # template
        for i in range(seg_p[1], seg_p[3]):
            prim[i] = 3 * np.ones(prim[i].size).reshape(-1, 1)  # np.random.randn(prim[i].size).reshape(-1, 1)

        # solve with dp
        mock_cache.cache_initial_state(prim[seg_p[1]])
        mock_cache.set_primal(prim)
        mock_cache.project_on_dynamics()
        dp_result, _ = mock_cache.get_primal()
        x_dp = np.asarray(dp_result[seg_p[1]: seg_p[2]])[:, :, 0]
        u_dp = np.asarray(dp_result[seg_p[2]: seg_p[3]])[:, :, 0]
        # ensure x0 stayed the same
        self.assertTrue(np.allclose(prim[seg_p[1]].T, x_dp[0]))

        # solve with cvxpy
        for i in range(seg_p[1], seg_p[3]):
            prim[i] = prim[i].reshape(-1,)

        x_bar = np.asarray(prim[seg_p[1]: seg_p[2]])
        u_bar = np.asarray(prim[seg_p[2]: seg_p[3]])
        N = self.__tree_from_markov.num_nodes
        n = self.__tree_from_markov.num_nonleaf_nodes
        x = cp.Variable(x_bar.shape)
        u = cp.Variable(u_bar.shape)
        # sum problem objectives and concatenate constraints
        cost = 0
        constraints = [x[0] == x_bar[0]]
        # nonleaf nodes
        for node in range(n):
            cost += cp.sum_squares(x[node] - x_bar[node]) + cp.sum_squares(u[node] - u_bar[node])
            for ch in self.__tree_from_markov.children_of(node):
                constraints += [x[ch] ==
                                self.__raocp_from_markov.state_dynamics_at_node(ch) @ x[node] +
                                self.__raocp_from_markov.control_dynamics_at_node(ch) @ u[node]]

        # leaf nodes
        for node in range(n, N):
            cost += cp.sum_squares(x[node] - x_bar[node])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        # ensure x0 stayed the same
        self.assertTrue(np.allclose(prim[seg_p[1]], x.value[0]))

        # check solutions are similar
        self.assertTrue(np.allclose(x.value, x_dp))
        self.assertTrue(np.allclose(u.value, u_dp))

    def test_kernel_projection(self):
        mock_cache, seg_p, _ = self._construct_mock_cache()
        _, prim = mock_cache.get_primal()  # template
        for i in range(seg_p[3], seg_p[6]):
            if i == seg_p[3] or i == seg_p[5]:
                pass
            else:
                prim[i] = np.random.randn(prim[i].size).reshape(-1, 1)

        mock_cache.set_primal(prim)
        mock_cache.project_on_kernel()
        proj, _ = mock_cache.get_primal()
        constraint_matrix = mock_cache.get_kernel_constraint_matrices()
        for i in range(self.__tree_from_markov.num_nonleaf_nodes):
            children = self.__tree_from_markov.children_of(i)
            t_stack = proj[seg_p[4] + children[0]]
            s_stack = proj[seg_p[5] + children[0]]
            if children.size > 1:
                for j in np.delete(children, 0):
                    t_stack = np.vstack((t_stack, proj[seg_p[4] + j]))
                    s_stack = np.vstack((s_stack, proj[seg_p[5] + j]))

            stack = np.vstack((proj[seg_p[3] + i], t_stack, s_stack))
            inf_norm = np.linalg.norm(constraint_matrix[i] @ stack, np.inf)
            self.assertTrue(np.allclose(inf_norm, 0))

    def test_modify_dual(self):
        mock_cache, _, seg_d = self._construct_mock_cache()
        parameter = 0
        while parameter == 0:
            parameter = np.linalg.norm(np.random.randn(1), 2)

        _, dual = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            dual[i] = np.random.randn(dual[i].size).reshape(-1, 1)

        mock_cache.set_dual(dual)
        mock_cache.modify_dual(parameter)
        new_dual, _ = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            test_result = dual[i] / parameter
            self.assertTrue(np.allclose(test_result, new_dual[i]))

    def test_add_halves(self):
        mock_cache, _, seg_d = self._construct_mock_cache()
        mock_cache.add_halves()
        dual, _ = mock_cache.get_dual()
        minus_half = -0.5
        plus_half = 0.5
        for i in dual[seg_d[5]: seg_d[6]]:
            self.assertEqual(i, minus_half)

        for i in dual[seg_d[6]: seg_d[7]]:
            self.assertEqual(i, plus_half)

        for i in dual[seg_d[12]: seg_d[13]]:
            self.assertEqual(i, minus_half)

        for i in dual[seg_d[13]: seg_d[14]]:
            self.assertEqual(i, plus_half)

    def test_modify_projection(self):
        mock_cache, _, seg_d = self._construct_mock_cache()
        _, zero_dual = mock_cache.get_dual()
        parameter = 0
        while parameter == 0:
            parameter = np.linalg.norm(np.random.randn(1), 2)

        _, dual = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            dual[i] = np.random.randn(dual[i].size).reshape(-1, 1)

        mock_cache.modify_projection(parameter, dual)
        new_dual, _ = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            test_result = parameter * (dual[i] - zero_dual[i])
            self.assertTrue(np.allclose(test_result, new_dual[i]))


if __name__ == '__main__':
    unittest.main()
