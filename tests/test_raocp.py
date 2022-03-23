import unittest
import raocp.core as rc
import numpy as np


class TestRAOCP(unittest.TestCase):
    __tree_from_markov = None
    __tree_from_iid = None
    __raocp_from_markov = None
    __raocp_from_iid = None

    @staticmethod
    def __construct_tree_from_markov():
        if TestRAOCP.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.5, 0])
            (N, tau) = (4, 3)
            TestRAOCP.__tree_from_markov = \
                rc.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

    @staticmethod
    def __construct_raocp_from_markov():
        if TestRAOCP.__raocp_from_markov is None:
            tree = TestRAOCP.__tree_from_markov

            Aw1 = np.eye(2)
            Aw2 = 2 * np.eye(2)
            Aw3 = 3 * np.eye(2)
            As = [Aw1, Aw2, Aw3]  # n x n matrices

            Bw1 = np.eye(2)
            Bw2 = 2 * np.eye(2)
            Bw3 = 3 * np.eye(2)
            Bs = [Bw1, Bw2, Bw3]  # n x u matrices

            cost_type = "quadratic"
            Q = 10 * np.eye(2)  # n x n matrix
            R = np.eye(2)  # u x u matrix OR scalar
            Pf = 5 * np.eye(2)  # n x n matrix

            (risk_type, alpha, cone_type, cone_dim) = ("AVaR", 0.5, "PosOrth", 3)

            TestRAOCP.__raocp_from_markov = rc.MarkovChainRAOCPProblemBuilder(scenario_tree=tree)\
                .with_possible_As_and_Bs(As, Bs)\
                .with_all_cost(cost_type, Q, R, Pf)\
                .with_all_risk(risk_type, alpha, cone_type, cone_dim)\
                .create()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRAOCP.__construct_tree_from_markov()
        TestRAOCP.__construct_raocp_from_markov()

    def test_A_B_at_node(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        Aw1 = np.eye(2)
        Aw2 = 2 * np.eye(2)
        Aw3 = 3 * np.eye(2)
        test_As = [Aw1, Aw2, Aw3]  # n x n matrices

        Bw1 = np.eye(2)
        Bw2 = 2 * np.eye(2)
        Bw3 = 3 * np.eye(2)
        test_Bs = [Bw1, Bw2, Bw3]  # n x u matrices

        num_nodes = tree.num_nodes()
        for i_node in range(1, num_nodes):
            w_value_at_node = tree.value_at_node(i_node)
            test_A_at_node = test_As[w_value_at_node]
            A_at_node = raocp.A_at_node(i_node)
            test_B_at_node = test_Bs[w_value_at_node]
            B_at_node = raocp.B_at_node(i_node)
            for row in range(test_A_at_node.shape[0]):
                for column in range(test_A_at_node.shape[1]):
                    self.assertEqual(test_A_at_node[row, column], A_at_node[row, column])
                    self.assertEqual(test_B_at_node[row, column], B_at_node[row, column])


if __name__ == '__main__':
    unittest.main()
