import unittest
import raocp.core as core
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
                core.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

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

            (risk_type, alpha) = ("AVaR", 0.5)

            TestRAOCP.__raocp_from_markov = core.MarkovChainRAOCPProblemBuilder(scenario_tree=tree) \
                .with_possible_As_and_Bs(As, Bs) \
                .with_all_cost(cost_type, Q, R, Pf) \
                .with_all_risk(risk_type, alpha) \
                .create()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRAOCP.__construct_tree_from_markov()
        TestRAOCP.__construct_raocp_from_markov()

    def test_A_at_node(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        Aw1 = np.eye(2)
        Aw2 = 2 * np.eye(2)
        Aw3 = 3 * np.eye(2)
        test_As = [Aw1, Aw2, Aw3]  # n x n matrices
        num_nodes = tree.num_nodes()
        for i_node in range(1, num_nodes):
            w_value_at_node = tree.value_at_node(i_node)
            test_A_at_node = test_As[w_value_at_node]
            A_at_node = raocp.A_at_node(i_node)
            np.testing.assert_array_equal(test_A_at_node, A_at_node)

    def test_B_at_node(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        Bw1 = np.eye(2)
        Bw2 = 2 * np.eye(2)
        Bw3 = 3 * np.eye(2)
        test_Bs = [Bw1, Bw2, Bw3]  # n x u matrices
        num_nodes = tree.num_nodes()
        for i_node in range(1, num_nodes):
            w_value_at_node = tree.value_at_node(i_node)
            test_B_at_node = test_Bs[w_value_at_node]
            B_at_node = raocp.B_at_node(i_node)
            np.testing.assert_array_equal(test_B_at_node, B_at_node)

    # Zichi, we need to talk about these tests ~ Ruairi ################################################################

    # def test_risk_values(self):
    #     tree = TestRAOCP.__tree_from_markov
    #     raocp = TestRAOCP.__raocp_from_markov
    #     for i_node in range(tree.num_nonleaf_nodes()):
    #         self.assertEqual("AVaR", raocp.risk_item_at_node(i_node).type)
    #         self.assertEqual(0.5, raocp.risk_item_at_node(i_node).alpha)
    #
    # def test_risk_E(self):
    #     tol = 1e-10
    #     tree = TestRAOCP.__tree_from_markov
    #     raocp = TestRAOCP.__raocp_from_markov
    #     alpha = 0.5
    #     for i_node in range(tree.num_nonleaf_nodes()):
    #         num_children = len(tree.conditional_probabilities_of_children(i_node))
    #         eye = np.eye(num_children)
    #         E_at_node = np.vstack((alpha * eye, -eye, np.ones((1, num_children))))
    #         for row in range(E_at_node.shape[0]):
    #             for column in range(E_at_node.shape[1]):
    #                 self.assertAlmostEqual(E_at_node[row, column],
    #                                        raocp.risk_item_at_node(i_node).E[row, column], delta=tol)
    #
    # def test_risk_F(self):
    #     tol = 1e-10
    #     tree = TestRAOCP.__tree_from_markov
    #     raocp = TestRAOCP.__raocp_from_markov
    #     for i_node in range(tree.num_nonleaf_nodes()):
    #         num_children = len(tree.conditional_probabilities_of_children(i_node))
    #         F_at_node = np.zeros((2 * num_children + 1, num_children))
    #         for row in range(F_at_node.shape[0]):
    #             for column in range(F_at_node.shape[1]):
    #                 self.assertAlmostEqual(F_at_node[row, column],
    #                                        raocp.risk_item_at_node(i_node).F[row, column], delta=tol)
    #
    # def test_risk_b(self):
    #     tol = 1e-10
    #     tree = TestRAOCP.__tree_from_markov
    #     raocp = TestRAOCP.__raocp_from_markov
    #     for i_node in range(tree.num_nonleaf_nodes()):
    #         num_children = len(tree.conditional_probabilities_of_children(i_node))
    #         pi = np.asarray(tree.conditional_probabilities_of_children(i_node)).reshape(num_children, 1)
    #         b_at_node = np.vstack((pi, np.zeros((num_children, 1)), 1))
    #         for row in range(b_at_node.shape[0]):
    #             self.assertAlmostEqual(b_at_node[row, 0], raocp.risk_item_at_node(i_node).b[row, 0], delta=tol)
    #
    # def test_risk_cone(self):
    #     tree = TestRAOCP.__tree_from_markov
    #     raocp = TestRAOCP.__raocp_from_markov
    #     cone_type = "NonnegOrth x NonnegOrth x Zero"
    #
    #     # create points for projection
    #     num_cones = 3
    #     num_samples = 100
    #     multiplier = 10
    #     x = [None] * num_cones
    #     cone_dim = 20
    #     samples = []
    #     for i in range(num_cones * 2):
    #         samples.append([None] * num_samples)
    #     for i in range(num_cones):
    #         x[i] = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
    #
    #     # create set samples
    #     for i in range(num_samples):
    #         samples[0][i] = np.random.randint(0, 100, cone_dim)  # non samples
    #         samples[1][i] = np.random.randint(0, 100, cone_dim)  # non samples
    #         samples[2][i] = np.zeros(cone_dim)  # zero samples
    #         samples[5][i] = np.zeros(cone_dim)  # zero dual samples (uni)
    #     samples[3] = samples[0]
    #     samples[4] = samples[1]
    #
    #     # test cartesian
    #     for i_node in range(tree.num_nonleaf_nodes()):
    #         self.assertEqual(cone_type, raocp.risk_item_at_node(i_node).cone.types)
    #         projection = raocp.risk_item_at_node(i_node).cone.project_onto_cone([x[0], x[1], x[2]])
    #         dual_projection = raocp.risk_item_at_node(i_node).cone.project_onto_cone([x[0], x[1], x[2]])
    #         for i in range(num_cones):
    #             for j in range(len(samples[0])):
    #                 self.assertTrue(np.inner((x[i].reshape((cone_dim,)) - projection[i].reshape((cone_dim,))),
    #                                          (samples[i][j].reshape((cone_dim,)) - projection[i].reshape(
    #                                              (cone_dim,)))) <= 0)
    #                 self.assertTrue(np.inner((x[i].reshape((cone_dim,)) - dual_projection[i].reshape((cone_dim,))),
    #                                          (samples[i + num_cones][j].reshape((cone_dim,)) - dual_projection[
    #                                              i].reshape(
    #                                              (cone_dim,)))) <= 0)


if __name__ == '__main__':
    unittest.main()
