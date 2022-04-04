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
            np.testing.assert_array_equal(test_A_at_node, A_at_node)
            np.testing.assert_array_equal(test_B_at_node, B_at_node)

    def test_cones_Uni(self):
        # create cone
        uni = core_cones.Uni()
        cones_type = "Uni"

        # create points for projection
        num_samples = 100
        multiplier = 10
        cone_dim = 20
        x = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
        samples = [None] * num_samples
        dual_samples = [None] * num_samples
        for i in range(num_samples):
            samples[i] = np.random.randint(-100, 100, 20)  # uni samples
            dual_samples[i] = np.zeros(cone_dim)  # uni dual samples (zero)

        # test uni
        self.assertEqual(cones_type, uni.type)
        projection = uni.project_onto_cone(x)
        dual_projection = uni.project_onto_dual(x)
        for i in range(len(samples)):
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - projection.reshape((cone_dim,)),
                                     samples[i].reshape((cone_dim,)) - projection.reshape((cone_dim,))) <= 0)
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - dual_projection.reshape((cone_dim,)),
                                     dual_samples[i].reshape((cone_dim,)) - dual_projection.reshape((cone_dim,))) <= 0)

    def test_cones_zero(self):
        # create cone
        zero = core_cones.Zero()
        cones_type = "Zero"

        # create points for projection
        num_samples = 100
        multiplier = 10
        cone_dim = 20
        x = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
        samples = [None] * num_samples
        dual_samples = [None] * num_samples
        for i in range(num_samples):
            samples[i] = np.zeros(cone_dim)  # zero samples
            dual_samples[i] = np.random.randint(-100, 100, 20)  # zero dual samples (uni)

        # test uni
        self.assertEqual(cones_type, zero.type)
        projection = zero.project_onto_cone(x)
        dual_projection = zero.project_onto_dual(x)
        for i in range(len(samples)):
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - projection.reshape((cone_dim,)),
                                     samples[i].reshape((cone_dim,)) - projection.reshape((cone_dim,))) <= 0)
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - dual_projection.reshape((cone_dim,)),
                                     dual_samples[i].reshape((cone_dim,)) - dual_projection.reshape((cone_dim,))) <= 0)

    def test_cones_non(self):
        # create cone
        non = core_cones.NonnegOrth()
        cones_type = "NonnegOrth"

        # create points for projection
        num_samples = 100
        multiplier = 10
        cone_dim = 20
        x = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
        samples = [None] * num_samples
        dual_samples = [None] * num_samples
        for i in range(num_samples):
            samples[i] = np.random.randint(0, 100, cone_dim)  # non samples
            dual_samples[i] = samples[i]

        # test uni
        self.assertEqual(cones_type, non.type)
        projection = non.project_onto_cone(x)
        dual_projection = non.project_onto_dual(x)
        for i in range(len(samples)):
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - projection.reshape((cone_dim,)),
                                     samples[i].reshape((cone_dim,)) - projection.reshape((cone_dim,))) <= 0)
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - dual_projection.reshape((cone_dim,)),
                                     dual_samples[i].reshape((cone_dim,)) - dual_projection.reshape((cone_dim,))) <= 0)

    def test_cones_soc(self):
        # create cone
        soc = core_cones.SOC()
        cones_type = "SOC"

        # create points for projection
        num_samples = 100
        multiplier = 10
        cone_dim = 20
        x = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
        samples = [None] * num_samples
        dual_samples = [None] * num_samples
        for i in range(num_samples):
            s = np.random.randint(-100, 100, cone_dim - 1)
            t = np.linalg.norm(s)
            samples[i] = (np.hstack((s, t)))  # soc samples
            dual_samples[i] = samples[i]

        # test uni
        self.assertEqual(cones_type, soc.type)
        projection = soc.project_onto_cone(x)
        dual_projection = soc.project_onto_dual(x)
        for i in range(len(samples)):
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - projection.reshape((cone_dim,)),
                                     samples[i].reshape((cone_dim,)) - projection.reshape((cone_dim,))) <= 0)
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - dual_projection.reshape((cone_dim,)),
                                     dual_samples[i].reshape((cone_dim,)) - dual_projection.reshape((cone_dim,))) <= 0)

    def test_cones_cart(self):
        # create cones
        uni = core_cones.Uni()
        zero = core_cones.Zero()
        non = core_cones.NonnegOrth()
        soc = core_cones.SOC()
        cones = [uni, zero, non, soc]
        cart = core_cones.Cart(cones)
        cart_type = "Uni x Zero x NonnegOrth x SOC"

        # create points for projection
        num_cones = len(cones)
        num_samples = 100
        multiplier = 10
        x = [None] * num_cones
        cone_dim = 20
        samples = []
        for i in range(num_cones * 2):
            samples.append([None] * num_samples)
        for i in range(num_cones):
            x[i] = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))

        # create set samples
        for i in range(num_samples):
            samples[0][i] = np.random.randint(-100, 100, 20)  # uni samples
            samples[1][i] = np.zeros(cone_dim)  # zero samples
            samples[2][i] = np.random.randint(0, 100, cone_dim)  # non samples
            s = np.random.randint(-100, 100, cone_dim - 1)
            t = np.linalg.norm(s)
            samples[3][i] = np.hstack((s, t))  # soc samples
            samples[4][i] = np.random.randint(-100, 100, cone_dim)  # uni dual samples (zero)
            samples[5][i] = np.zeros(cone_dim)  # zero dual samples (uni)
        samples[6] = samples[2]
        samples[7] = samples[3]

        # test cartesian
        self.assertEqual(cart_type, cart.type)
        projection = cart.project_onto_cone([x[0], x[1], x[2], x[3]])
        dual_projection = cart.project_onto_cone([x[0], x[1], x[2], x[3]])
        for i in range(num_cones):
            for j in range(len(samples[0])):
                self.assertTrue(np.inner((x[i].reshape((cone_dim,)) - projection[i].reshape((cone_dim,))),
                                         (samples[i][j].reshape((cone_dim,)) - projection[i].reshape(
                                             (cone_dim,)))) <= 0)
                self.assertTrue(np.inner((x[i].reshape((cone_dim,)) - dual_projection[i].reshape((cone_dim,))),
                                         (samples[i+num_cones][j].reshape((cone_dim,)) - dual_projection[i].reshape(
                                             (cone_dim,)))) <= 0)

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

    def test_risk_values(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i_node in range(tree.num_nonleaf_nodes()):
            self.assertEqual("AVaR", raocp.risk_item_at_node(i_node).type)
            self.assertEqual(0.5, raocp.risk_item_at_node(i_node).alpha)
            self.assertEqual("NonnegOrth x NonnegOrth x Zero", raocp.risk_item_at_node(i_node).cone.type)

    def test_risk_E(self):
        tol = 1e-10
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
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
        for i_node in range(tree.num_nonleaf_nodes()):
            self.assertEqual("AVaR", raocp.risk_item_at_node(i_node).type)
            self.assertEqual(0.5, raocp.risk_item_at_node(i_node).alpha)
            self.assertEqual("NonnegOrth x NonnegOrth x Zero", raocp.risk_item_at_node(i_node).cone.type)
            for row in range(E[i_node].shape[0]):
                for column in range(E[i_node].shape[1]):
                    self.assertAlmostEqual(E[i_node][row, column],
                                           raocp.risk_item_at_node(i_node).E[row, column], delta=tol)

    def test_risk_b(self):
        tol = 1e-10
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
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
        for i_node in range(tree.num_nonleaf_nodes()):
            for row in range(b[i_node].shape[0]):
                self.assertAlmostEqual(b[i_node][row, 0], raocp.risk_item_at_node(i_node).b[row, 0], delta=tol)


if __name__ == '__main__':
    unittest.main()
