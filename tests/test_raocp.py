import unittest
import raocp.core.scenario_tree as core_tree
import raocp.core.problem_spec as core_spec
import numpy as np


class TestRAOCP(unittest.TestCase):
    __tree_from_markov = None
    __tree_from_iid = None
    __raocp_from_markov = None
    __raocp_from_iid = None
    __good_size = 3

    @staticmethod
    def __construct_tree_from_markov():
        if TestRAOCP.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.5, 0])
            (N, tau) = (4, 3)
            TestRAOCP.__tree_from_markov = \
                core_tree.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

    @staticmethod
    def __construct_raocp_from_markov():
        if TestRAOCP.__raocp_from_markov is None:
            tree = TestRAOCP.__tree_from_markov

            # construct markovian set of system and control dynamics
            system = np.eye(2)
            set_system = [system, 2 * system, 3 * system]  # n x n matrices
            control = np.eye(2)
            set_control = [control, 2 * control, 3 * control]  # n x u matrices

            # construct cost weight matrices
            cost_type = "Quadratic"
            nonleaf_state_weight = 10 * np.eye(2)  # n x n matrix
            control_weight = np.eye(2)  # u x u matrix OR scalar
            leaf_state_weight = 5 * np.eye(2)  # n x n matrix

            # define risks
            (risk_type, alpha) = ("AVaR", 0.5)

            # create problem
            TestRAOCP.__raocp_from_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_dynamics(set_system, set_control) \
                .with_all_costs(cost_type, nonleaf_state_weight, control_weight, leaf_state_weight) \
                .with_all_risks(risk_type, alpha)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRAOCP.__construct_tree_from_markov()
        TestRAOCP.__construct_raocp_from_markov()

    def test_markovian_system_dynamics_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        self.assertTrue(raocp.list_of_system_dynamics[0] is None)
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_system_dynamics[i] is not None)

    def test_markovian_control_dynamics_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        self.assertTrue(raocp.list_of_control_dynamics[0] is None)
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_control_dynamics[i] is not None)

    def test_cost_items_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(tree.num_nodes):
            self.assertTrue(raocp.list_of_cost_items[i] is not None)

    def test_risk_items_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(tree.num_nonleaf_nodes):
            self.assertTrue(raocp.list_of_risk_items[i] is not None)

    def test_markovian_system_dynamics_failure(self):
        tree = TestRAOCP.__tree_from_markov

        # construct bad markovian set of system dynamics
        set_system = [np.eye(TestRAOCP.__good_size + 1), np.eye(TestRAOCP.__good_size)]  # n x n matrices
        # construct good markovian set of control dynamics
        control = np.ones((TestRAOCP.__good_size, 1))
        set_control = [control, 2 * control]  # n x u matrices

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_markovian_dynamics(set_system, set_control)

    def test_markovian_control_dynamics_failure(self):
        tree = TestRAOCP.__tree_from_markov

        # construct good markovian set of system dynamics
        system = np.eye(TestRAOCP.__good_size)
        set_system = [system, 2 * system]  # n x n matrices
        # construct bad markovian set of control dynamics
        set_control = [np.ones((TestRAOCP.__good_size + 1, 1)), np.ones((TestRAOCP.__good_size, 1))]  # n x u matrices

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_markovian_dynamics(set_system, set_control)

    def test_markovian_system_and_control_dynamics_failure(self):
        tree = TestRAOCP.__tree_from_markov

        # construct good markovian set of system dynamics (rows = 3)
        system = np.eye(TestRAOCP.__good_size)
        set_system = [system, 2 * system]  # n x n matrices
        # construct good markovian set of control dynamics  (rows = 4)
        control = np.ones((TestRAOCP.__good_size + 1, 1))
        set_control = [control, 2 * control]  # n x u matrices

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_markovian_dynamics(set_system, set_control)

    def test_cost_items_nonleaf_state_failure(self):
        tree = TestRAOCP.__tree_from_markov
        cost_type = "Quadratic"

        # construct bad nonleaf state weights
        state_weights = np.eye(TestRAOCP.__good_size + 1)

        # construct good control and terminal state weights
        control_weights = np.ones((TestRAOCP.__good_size, 1))
        terminal_state_weights = np.eye(TestRAOCP.__good_size)

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_all_costs(cost_type, state_weights, control_weights, terminal_state_weights)

    def test_cost_items_control_failure(self):
        tree = TestRAOCP.__tree_from_markov
        cost_type = "Quadratic"

        # construct bad control weights
        control_weights = np.ones((TestRAOCP.__good_size + 1, 1))

        # construct good control and terminal state weights
        state_weights = np.eye(TestRAOCP.__good_size)
        terminal_state_weights = np.eye(TestRAOCP.__good_size)

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_all_costs(cost_type, state_weights, control_weights, terminal_state_weights)

    def test_cost_items_leaf_state_failure(self):
        tree = TestRAOCP.__tree_from_markov
        cost_type = "Quadratic"

        # construct bad leaf state weights
        terminal_state_weights = np.eye(TestRAOCP.__good_size + 1)

        # construct good control and state weights
        control_weights = np.ones((TestRAOCP.__good_size, 1))
        state_weights = np.eye(TestRAOCP.__good_size)

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_all_costs(cost_type, state_weights, control_weights, terminal_state_weights)

    def test_cost_items_states_failure(self):
        tree = TestRAOCP.__tree_from_markov
        cost_type = "Quadratic"

        # construct medium bad nonleaf and leaf state weights (same # of columns, different # of rows)
        state_weights = np.ones((TestRAOCP.__good_size, TestRAOCP.__good_size))
        terminal_state_weights = np.ones((TestRAOCP.__good_size + 1, TestRAOCP.__good_size))

        # construct good control weights
        control_weights = np.ones((TestRAOCP.__good_size, 1))

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_all_costs(cost_type, state_weights, control_weights, terminal_state_weights)


if __name__ == '__main__':
    unittest.main()
