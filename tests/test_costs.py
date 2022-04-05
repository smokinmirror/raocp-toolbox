import unittest
import raocp.core.costs as core_costs
import numpy as np


class TestCosts(unittest.TestCase):
    # __num_samples = 100

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_cost_value(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        x0 = np.array([[1],
                       [1]])

        def x_all(initial_state):
            x_list = [initial_state]
            for i in range(1, tree.num_nodes()):
                node = tree.ancestor_of(i)
                x_at_node = raocp.A_at_node(i) @ x_list[node]
                x_list.append(x_at_node)
            return x_list

        cost_type = "quadratic"
        cost = [10.0, 10.0, 40.0, 10.0, 40.0, 90.0, 40.0, 160.0, 10.0, 40.0, 90.0, 40.0, 160.0, 360.0, 810.0, 40.0,
                160.0, 360.0, 160.0, 640.0, 10.0, 160.0, 810.0, 40.0, 640.0, 1440.0, 7290.0, 40.0, 640.0, 3240.0, 160.0,
                2560.0]
        for i_node in range(tree.num_nodes()):
            self.assertEqual(cost_type, raocp.cost_item_at_node(i_node).type)
            self.assertEqual(cost[i_node], raocp.cost_item_at_node(i_node).get_cost(x_all(x0)[i_node]))

    def test_cost_Q(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        Q = 10 * np.eye(2)  # n x n matrix
        for i_node in range(tree.num_nodes()):
            for row in range(Q.shape[0]):
                for column in range(Q.shape[1]):
                    self.assertEqual(Q[row, column], raocp.cost_item_at_node(i_node).Q[row, column])

    def test_cost_R(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        R = np.eye(2)  # u x u matrix OR scalar
        for i_node in range(tree.num_nodes()):
            for row in range(R.shape[0]):
                for column in range(R.shape[1]):
                    self.assertEqual(R[row, column], raocp.cost_item_at_node(i_node).R[row, column])

    def test_cost_Pf(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        Pf = 5 * np.eye(2)  # n x n matrix
        for i_node in range(tree.num_nodes()):
            for row in range(Pf.shape[0]):
                for column in range(Pf.shape[1]):
                    self.assertEqual(Pf[row, column], raocp.cost_item_at_node(i_node).Pf[row, column])