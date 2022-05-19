import unittest
import numpy as np
import raocp.core.scenario_tree as core_tree
import raocp.core.problem_spec as core_spec
import raocp.core.cache as core_cache
import raocp.core.operators as core_operators


class TestOperators(unittest.TestCase):
    __tree_from_markov = None
    __raocp_from_markov = None
    __operator_from_raocp = None
    __good_size = 3

    @staticmethod
    def _construct_tree_from_markov():
        if TestOperators.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.5, 0])
            (N, tau) = (4, 3)
            TestOperators.__tree_from_markov = core_tree.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

    @staticmethod
    def _construct_raocp_from_markov():
        if TestOperators.__raocp_from_markov is None:
            tree = TestOperators.__tree_from_markov

            # construct markovian set of system and control dynamics
            system = np.eye(TestOperators.__good_size)
            set_system = [system, 2 * system, 3 * system]  # n x n matrices
            control = np.eye(TestOperators.__good_size)
            set_control = [control, 2 * control, 3 * control]  # n x u matrices

            # construct cost weight matrices
            cost_type = "Quadratic"
            cost_types = [cost_type] * TestOperators.__good_size
            nonleaf_state_weight = 10 * np.eye(TestOperators.__good_size)  # n x n matrix
            nonleaf_state_weights = [nonleaf_state_weight, 2 * nonleaf_state_weight, 3 * nonleaf_state_weight]
            control_weight = np.eye(TestOperators.__good_size)  # u x u matrix OR scalar
            control_weights = [control_weight, 2 * control_weight, 3 * control_weight]
            leaf_state_weight = 5 * np.eye(TestOperators.__good_size)  # n x n matrix

            # define risks
            (risk_type, alpha) = ("AVaR", 0.5)

            TestOperators.__raocp_from_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_dynamics(set_system, set_control) \
                .with_markovian_costs(cost_types, nonleaf_state_weights, control_weights) \
                .with_all_leaf_costs(cost_type, leaf_state_weight) \
                .with_all_risks(risk_type, alpha)

    @staticmethod
    def _construct_operators_from_raocp():
        if TestOperators.__operator_from_raocp is None:
            num_nonleaf_nodes = TestOperators.__tree_from_markov.__num_nonleaf_nodes
            num_nodes = TestOperators.__tree_from_markov.__num_nodes
            num_stages = TestOperators.__tree_from_markov.__num_stages
            primal_split = [0,
                            num_nodes,
                            num_nodes + num_nonleaf_nodes * 1,
                            num_nodes + num_nonleaf_nodes * 2,
                            num_nodes + num_nonleaf_nodes * 2 + (num_stages + 1),
                            num_nodes + num_nonleaf_nodes * 2 + (num_stages + 1) * 2]
            dual_split = [0,
                          num_nonleaf_nodes * 1,
                          num_nonleaf_nodes * 2,
                          num_nonleaf_nodes * 2 + num_nodes * 1,
                          num_nonleaf_nodes * 2 + num_nodes * 2,
                          num_nonleaf_nodes * 2 + num_nodes * 3,
                          num_nonleaf_nodes * 2 + num_nodes * 4,
                          num_nonleaf_nodes * 2 + num_nodes * 5,
                          num_nonleaf_nodes * 2 + num_nodes * 6,
                          num_nonleaf_nodes * 2 + num_nodes * 7]
            TestOperators.__operator_from_raocp = core_operators.Operator(
                TestOperators.__raocp_from_markov, primal_split, dual_split)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestOperators._construct_tree_from_markov()
        TestOperators._construct_raocp_from_markov()
        TestOperators._construct_operators_from_raocp()

    def test_ell(self):
        pass
        # random_primal = np.random.


if __name__ == '__main__':
    unittest.main()
