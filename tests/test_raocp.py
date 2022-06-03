import unittest
import numpy as np
import raocp.core.costs as core_costs
import raocp.core.dynamics as core_dynamics
import raocp.core.nodes as core_nodes
import raocp.core.raocp_spec as core_spec
import raocp.core.risks as core_risks
import raocp.core.scenario_tree as core_tree
import raocp.core.constraints.rectangle as rectangle


class TestRAOCP(unittest.TestCase):
    __tree_from_markov = None
    __tree_from_iid = None
    __raocp_from_markov = None
    __raocp_from_markov_with_markov = None
    __raocp_from_iid = None
    __good_size = 3

    @staticmethod
    def _construct_tree_from_markov():
        if TestRAOCP.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.4, 0.1])
            (N, tau) = (4, 3)
            TestRAOCP.__tree_from_markov = \
                core_tree.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

    @staticmethod
    def _construct_raocp_from_markov():
        if TestRAOCP.__raocp_from_markov is None:
            tree = TestRAOCP.__tree_from_markov

            # construct markovian set of system and control dynamics
            system = np.eye(2)
            set_system = [system, 2 * system, 3 * system]  # n x n matrices
            control = np.eye(2)
            set_control = [control, 2 * control, 3 * control]  # n x u matrices
            dynamics = [core_dynamics.Dynamics(set_system[0], set_control[0]),
                        core_dynamics.Dynamics(set_system[1], set_control[1]),
                        core_dynamics.Dynamics(set_system[2], set_control[2])]

            # construct cost weight matrices
            nl = core_nodes.Nonleaf()
            l = core_nodes.Leaf()
            nonleaf_state_weight = 10 * np.eye(2)  # n x n matrix
            nonleaf_state_weights = [nonleaf_state_weight, 2 * nonleaf_state_weight, 3 * nonleaf_state_weight]
            control_weight = np.eye(2)  # u x u matrix OR scalar
            control_weights = [control_weight, 2 * control_weight, 3 * control_weight]
            nonleaf_cost = core_costs.Quadratic(nl, nonleaf_state_weight, control_weight)
            nonleaf_costs = [core_costs.Quadratic(nl, nonleaf_state_weights[0], control_weights[0]),
                             core_costs.Quadratic(nl, nonleaf_state_weights[1], control_weights[1]),
                             core_costs.Quadratic(nl, nonleaf_state_weights[2], control_weights[2])]
            leaf_state_weight = 5 * np.eye(2)  # n x n matrix
            leaf_costs = core_costs.Quadratic(l, leaf_state_weight)

            # construct constraint min and max
            nonleaf_size = TestRAOCP.__good_size + TestRAOCP.__good_size
            leaf_size = TestRAOCP.__good_size
            nl_min = -2 * np.ones((nonleaf_size, 1))
            nl_max = 2 * np.ones((nonleaf_size, 1))
            l_min = -0.5 * np.ones((leaf_size, 1))
            l_max = 0.5 * np.ones((leaf_size, 1))
            nl_rect = rectangle.Rectangle(nl, nl_min, nl_max)
            l_rect = rectangle.Rectangle(l, l_min, l_max)

            # define risks
            alpha = 0.5
            risks = core_risks.AVaR(alpha)

            # create problem
            TestRAOCP.__raocp_from_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_dynamics(dynamics) \
                .with_all_nonleaf_costs(nonleaf_cost) \
                .with_all_leaf_costs(leaf_costs) \
                .with_all_risks(risks)

            TestRAOCP.__raocp_from_markov_with_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_dynamics(dynamics) \
                .with_markovian_nonleaf_costs(nonleaf_costs) \
                .with_all_leaf_costs(leaf_costs) \
                .with_markovian_nonleaf_constraints([nl_rect, nl_rect, nl_rect]) \
                .with_all_leaf_constraints(l_rect) \
                .with_all_risks(risks)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRAOCP._construct_tree_from_markov()
        TestRAOCP._construct_raocp_from_markov()

    def test_markovian_dynamics_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov_with_markov
        self.assertTrue(raocp.list_of_dynamics[0] is None)
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_dynamics[i] is not None)

    def test_markovian_nonleaf_costs_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov_with_markov
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_nonleaf_costs[i] is not None)

    def test_all_nonleaf_costs_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_nonleaf_costs[i] is not None)

    def test_leaf_costs_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(tree.num_nodes):
            if i < tree.num_nonleaf_nodes:
                self.assertTrue(raocp.list_of_leaf_costs[i] is None)
            else:
                self.assertTrue(raocp.list_of_leaf_costs[i] is not None)

    def test_no_constraints_loaded(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov_with_markov = core_spec.RAOCP(scenario_tree=tree)
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_nonleaf_constraints[i] is not None)
            if i >= tree.num_nonleaf_nodes:
                self.assertTrue(raocp.list_of_leaf_constraints[i] is not None)

    def test_markovian_nonleaf_constraints_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov_with_markov
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_nonleaf_constraints[i] is not None)

    def test_all_nonleaf_constraints_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_nonleaf_constraints[i] is not None)

    def test_leaf_constraints_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(tree.num_nodes):
            if i < tree.num_nonleaf_nodes:
                self.assertTrue(raocp.list_of_leaf_constraints[i] is None)
            else:
                self.assertTrue(raocp.list_of_leaf_constraints[i] is not None)

    def test_risks_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(tree.num_nonleaf_nodes):
            self.assertTrue(raocp.list_of_risks[i] is not None)

    def test_markovian_dynamics_failure(self):
        tree = TestRAOCP.__tree_from_markov
        bad = [core_dynamics.Dynamics(np.eye(TestRAOCP.__good_size + 1), np.ones((TestRAOCP.__good_size + 1, 1))),
               core_dynamics.Dynamics(np.eye(TestRAOCP.__good_size), np.ones((TestRAOCP.__good_size, 1)))]

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_markovian_dynamics(bad)

    def test_constraints_before_dynamics_markovian(self):
        tree = TestRAOCP.__tree_from_markov
        nl = core_nodes.Nonleaf()

        # construct markovian set of system and control dynamics
        system = np.eye(2)
        control = np.eye(2)
        dynamics = [core_dynamics.Dynamics(system, control),
                    core_dynamics.Dynamics(system, control),
                    core_dynamics.Dynamics(system, control)]

        # construct constraint min and max
        nonleaf_size = TestRAOCP.__good_size + TestRAOCP.__good_size
        nl_min = -2 * np.ones((nonleaf_size, 1))
        nl_max = 2 * np.ones((nonleaf_size, 1))
        nl_rect = rectangle.Rectangle(nl, nl_min, nl_max)

        with self.assertRaises(Exception):
            TestRAOCP.__raocp_from_markov_with_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_nonleaf_constraints([nl_rect, nl_rect, nl_rect]) \
                .with_markovian_dynamics(dynamics)

    def test_constraints_before_dynamics_nonleaf(self):
        tree = TestRAOCP.__tree_from_markov
        nl = core_nodes.Nonleaf()

        # construct markovian set of system and control dynamics
        system = np.eye(2)
        control = np.eye(2)
        dynamics = [core_dynamics.Dynamics(system, control),
                    core_dynamics.Dynamics(system, control),
                    core_dynamics.Dynamics(system, control)]

        # construct constraint min and max
        nonleaf_size = TestRAOCP.__good_size + TestRAOCP.__good_size
        nl_min = -2 * np.ones((nonleaf_size, 1))
        nl_max = 2 * np.ones((nonleaf_size, 1))
        nl_rect = rectangle.Rectangle(nl, nl_min, nl_max)

        with self.assertRaises(Exception):
            TestRAOCP.__raocp_from_markov_with_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_all_nonleaf_constraints(nl_rect) \
                .with_markovian_dynamics(dynamics)

    def test_constraints_before_dynamics_leaf(self):
        tree = TestRAOCP.__tree_from_markov
        l = core_nodes.Leaf()

        # construct markovian set of system and control dynamics
        system = np.eye(2)
        control = np.eye(2)
        dynamics = [core_dynamics.Dynamics(system, control),
                    core_dynamics.Dynamics(system, control),
                    core_dynamics.Dynamics(system, control)]

        # construct constraint min and max
        leaf_size = TestRAOCP.__good_size
        l_min = -0.5 * np.ones((leaf_size, 1))
        l_max = 0.5 * np.ones((leaf_size, 1))
        l_rect = rectangle.Rectangle(l, l_min, l_max)

        with self.assertRaises(Exception):
            TestRAOCP.__raocp_from_markov_with_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_all_leaf_constraints(l_rect) \
                .with_markovian_dynamics(dynamics)


if __name__ == '__main__':
    unittest.main()
