import unittest
import raocp.core as rc
import numpy as np
import raocp.core.cones as core_cones


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

            Aw1 = np.eye(3)
            Aw2 = 2 * np.eye(3)
            Aw3 = 3 * np.eye(3)
            As = [Aw1, Aw2, Aw3]  # n x n matrices

            Bw1 = np.eye(2)
            Bw2 = 2 * np.eye(2)
            Bw3 = 3 * np.eye(2)
            Bs = [Bw1, Bw2, Bw3]  # n x u matrices

            cost_type = "quadratic"
            Q = 5 * np.eye(3)  # n x n matrix
            R = np.eye(2)  # u x u matrix OR scalar
            Pf = 10 * np.eye(3)  # n x n matrix

            (risk_type, alpha) = ("AVaR", 0.5)

            TestRAOCP.__raocp_from_markov = rc.MarkovChainRAOCPProblemBuilder(scenario_tree=tree) \
                .with_possible_As_and_Bs(As, Bs) \
                .with_all_cost(cost_type, Q, R, Pf) \
                .with_all_risk(risk_type, alpha) \
                .create()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRAOCP.__construct_tree_from_markov()
        TestRAOCP.__construct_raocp_from_markov()

    def test_A_B_at_node(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        Aw1 = np.eye(3)
        Aw2 = 2 * np.eye(3)
        Aw3 = 3 * np.eye(3)
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
            np.testing.assert_array_equal(test_A_at_node, A_at_node)
            np.testing.assert_array_equal(test_B_at_node, B_at_node)

    def test_cones(self):
        tol = 1e-10
        # create cones
        uni = core_cones.Uni()
        zero = core_cones.Zero()
        non = core_cones.NonnegOrth()
        soc = core_cones.SOC()
        cones = [uni, zero, non, soc]
        cones_type = ["Uni", "Zero", "NonnegOrth", "SOC"]
        cart = core_cones.Cart(cones)
        cart_type = "Uni x Zero x NonnegOrth x SOC"

        # create points for projection
        num_cones = len(cones)
        multiplier = 10
        x = [] * num_cones
        cone_dim = [] * num_cones
        samples = [] * num_cones
        projection = [] * num_cones
        dual_projection = [] * num_cones
        for i in range(num_cones):
            cone_dim[i] = np.random.randint(2, 20)
            x[i] = multiplier * np.random.rand(cone_dim[i])

        # create set samples
        for i in range(100):
            samples[0].append(np.random.randint(-100, 100, x[0].size))  # uni samples
            samples[1].append(np.zeros(x[1].size))  # zero samples
            samples[2].append(np.random.randint(0, 100, x[2].size))  # non samples
            s = np.random.randint(-100, 100, x[3].size-1)
            t = np.linalg.norm(s)
            samples[3].append(np.concatenate((s, t)))  # soc samples
            samples[4].append(np.random.randint(-100, 100, x[1].size))  # uni dual samples (non)
            samples[5].append(np.zeros(x[0].size))  # zero dual samples (uni)
            samples[6] = samples[2]
            samples[7] = samples[3]

        # test cones
        for i in range(num_cones):
            self.assertEqual(cones_type[i], cones[i].type)
            projection[i] = cones[i].project_onto_cone(x[i])
            dual_projection[i] = cones[i].project_onto_dual(x[i])
            for j in range(samples[i].size):
                self.assertTrue(np.inner(projection[i], samples[i]))
                self.assertTrue(np.inner(projection[i], samples[i+4]))

        # # test cartesian
        # for i_cones in range(len(cones_type)):
        #     self.assertEqual(cart_type, cart.type)
        #
        #     self.assertEqual(20, cart.dimension)
        # for i_cones in range(len(cones_type)):
        #     self.assertEqual(cones_dimension[i_cones], cart.dimensions[i_cones])

    def test_cost(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov

        # create test examples
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
        Q = 10 * np.eye(2)  # n x n matrix
        R = np.eye(2)  # u x u matrix OR scalar
        Pf = 5 * np.eye(2)  # n x n matrix
        cost = [10.0, 10.0, 40.0, 10.0, 40.0, 90.0, 40.0, 160.0, 10.0, 40.0, 90.0, 40.0, 160.0, 360.0, 810.0, 40.0,
                160.0, 360.0, 160.0, 640.0, 10.0, 160.0, 810.0, 40.0, 640.0, 1440.0, 7290.0, 40.0, 640.0, 3240.0, 160.0,
                2560.0]

        # test cost
        for i_node in range(tree.num_nodes()):
            self.assertEqual(cost_type, raocp.cost_item_at_node(i_node).type)
            self.assertEqual(cost[i_node], raocp.cost_item_at_node(i_node).get_cost(x_all(x0)[i_node]))
            for row in range(Q.shape[0]):
                for column in range(Q.shape[1]):
                    self.assertEqual(Q[row, column], raocp.cost_item_at_node(i_node).Q[row, column])
            for row in range(R.shape[0]):
                for column in range(R.shape[1]):
                    self.assertEqual(R[row, column], raocp.cost_item_at_node(i_node).R[row, column])
            for row in range(Pf.shape[0]):
                for column in range(Pf.shape[1]):
                    self.assertEqual(Pf[row, column], raocp.cost_item_at_node(i_node).Pf[row, column])

    def test_risk(self):
        tol = 1e-10
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov

        # create test examples
        E = [np.array([[0.5, 0.], [0., 0.5], [-1., -0.], [-0., -1.], [1., 1.]]),
             np.array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5], [-1., -0., -0.], [-0., -1., -0.], [-0., -0., -1.],
                       [1., 1., 1.]]),
             np.array([[0.5, 0.], [0., 0.5], [-1., -0.], [-0., -1.], [1., 1.]]),
             np.array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5], [-1., -0., -0.], [-0., -1., -0.], [-0., -0., -1.],
                       [1., 1., 1.]]),
             np.array([[0.5, 0.], [0., 0.5], [-1., -0.], [-0., -1.], [1., 1.]]),
             np.array([[0.5, 0.], [0., 0.5], [-1., -0.], [-0., -1.], [1., 1.]]),
             np.array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5], [-1., -0., -0.], [-0., -1., -0.], [-0., -0., -1.],
                       [1., 1., 1.]]),
             np.array([[0.5, 0.], [0., 0.5], [-1., -0.], [-0., -1.], [1., 1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]]),
             np.array([[0.5], [-1.], [1.]])]
        b = [np.array([[0.5], [0.5], [0.], [0.], [1.]]),
             np.array([[0.1], [0.8], [0.1], [0.], [0.], [0.], [1.]]),
             np.array([[0.4], [0.6], [0.], [0.], [1.]]),
             np.array([[0.1], [0.8], [0.1], [0.], [0.], [0.], [1.]]),
             np.array([[0.4], [0.6], [0.], [0.], [1.]]),
             np.array([[0.3], [0.7], [0.], [0.], [1.]]),
             np.array([[0.1], [0.8], [0.1], [0.], [0.], [0.], [1.]]),
             np.array([[0.4], [0.6], [0.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]]),
             np.array([[1.], [0.], [1.]])]

        # test risk
        for i_node in range(tree.num_nonleaf_nodes()):
            self.assertEqual("AVaR", raocp.risk_item_at_node(i_node).type)
            self.assertEqual(0.5, raocp.risk_item_at_node(i_node).alpha)
            self.assertEqual("NonnegOrth x NonnegOrth x Zero", raocp.risk_item_at_node(i_node).cone.type)
            for row in range(E[i_node].shape[0]):
                for column in range(E[i_node].shape[1]):
                    self.assertAlmostEqual(E[i_node][row, column],
                                           raocp.risk_item_at_node(i_node).E[row, column], delta=tol)
            for row in range(b[i_node].shape[0]):
                self.assertAlmostEqual(b[i_node][row, 0], raocp.risk_item_at_node(i_node).b[row, 0], delta=tol)


if __name__ == '__main__':
    unittest.main()
