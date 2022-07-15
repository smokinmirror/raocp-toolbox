import unittest
import numpy as np
import raocp.core.costs as core_costs
import raocp.core.nodes as core_nodes
from scipy.linalg import sqrtm


class TestCosts(unittest.TestCase):
    __QuadraticNonleaf = None
    __QuadraticLeaf = None
    __state = np.array([[-2], [2], [4]])
    __control = np.array([[5], [-5], [2]])
    __size = __state.size
    __multipliers = [2, 3, 4]
    (__nl, __l) = core_nodes.Nonleaf(), core_nodes.Leaf()

    @staticmethod
    def _construct_classes():
        nonleaf_state_weights = TestCosts.__multipliers[0] * np.eye(TestCosts.__size)
        control_weights = TestCosts.__multipliers[1] * np.eye(TestCosts.__size)
        leaf_state_weights = TestCosts.__multipliers[2] * np.eye(TestCosts.__size)
        TestCosts.__QuadraticNonleaf = core_costs.Quadratic(TestCosts.__nl, nonleaf_state_weights, control_weights)
        TestCosts.__QuadraticLeaf = core_costs.Quadratic(TestCosts.__l, leaf_state_weights)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestCosts._construct_classes()

    def test_quadratic_nonleaf_check_control_weights(self):
        weights = np.ones((TestCosts.__size, 1))
        TestCosts.__QuadraticNonleaf._check_control_weights(weights)

    def test_quadratic_nonleaf_check_control_weights_failure(self):
        with self.assertRaises(Exception):
            TestCosts.__QuadraticNonleaf._check_control_weights(None)

    def test_quadratic_leaf_check_control_weights(self):
        TestCosts.__QuadraticLeaf._check_control_weights(None)

    def test_quadratic_leaf_check_control_weights_failure(self):
        weights = np.ones((TestCosts.__size, 1))
        with self.assertRaises(Exception):
            TestCosts.__QuadraticLeaf._check_control_weights(weights)

    def test_non_square_state_weights_nonleaf(self):
        ns_weights = np.ones((TestCosts.__size, TestCosts.__size + 1))
        s_weights = np.ones((TestCosts.__size, TestCosts.__size))
        with self.assertRaises(Exception):
            core_costs.Quadratic(TestCosts.__nl, ns_weights, s_weights)

    def test_non_square_control_weights_nonleaf(self):
        ns_weights = np.ones((TestCosts.__size, TestCosts.__size + 1))
        s_weights = np.ones((TestCosts.__size, TestCosts.__size))
        with self.assertRaises(Exception):
            core_costs.Quadratic(TestCosts.__nl, s_weights, ns_weights)

    def test_non_square_state_weights_leaf(self):
        ns_weights = np.ones((TestCosts.__size, TestCosts.__size + 1))
        with self.assertRaises(Exception):
            core_costs.Quadratic(TestCosts.__l, ns_weights)

    def test_state_matrix_getter_nonleaf(self):
        dud = np.zeros((TestCosts.__size, TestCosts.__size))
        weights = np.ones((TestCosts.__size, TestCosts.__size))
        mock_cost = core_costs.Quadratic(TestCosts.__nl, weights, dud)
        self.assertTrue(np.array_equal(mock_cost.state_weights, weights))

    def test_control_matrix_getter_nonleaf(self):
        dud = np.zeros((TestCosts.__size, TestCosts.__size))
        weights = np.ones((TestCosts.__size, TestCosts.__size))
        mock_cost = core_costs.Quadratic(TestCosts.__nl, dud, weights)
        self.assertTrue(np.array_equal(mock_cost.control_weights, weights))

    def test_state_matrix_getter_leaf(self):
        weights = np.ones((TestCosts.__size, TestCosts.__size))
        mock_cost = core_costs.Quadratic(TestCosts.__l, weights)
        self.assertTrue(np.array_equal(mock_cost.state_weights, weights))

    def test_sqrt_state_matrix_getter_nonleaf(self):
        dud = np.zeros((TestCosts.__size, TestCosts.__size))
        weights = 10 * np.random.sample(1) * np.eye(TestCosts.__size)
        mock_cost = core_costs.Quadratic(TestCosts.__nl, weights, dud)
        self.assertTrue(np.array_equal(mock_cost.sqrt_state_weights, sqrtm(weights)))

    def test_sqrt_control_matrix_getter_nonleaf(self):
        dud = np.zeros((TestCosts.__size, TestCosts.__size))
        weights = 10 * np.random.sample(1) * np.eye(TestCosts.__size)
        mock_cost = core_costs.Quadratic(TestCosts.__nl, dud, weights)
        self.assertTrue(np.array_equal(mock_cost.sqrt_control_weights, sqrtm(weights)))

    def test_sqrt_state_matrix_getter_leaf(self):
        weights = 10 * np.random.sample(1) * np.eye(TestCosts.__size)
        mock_cost = core_costs.Quadratic(TestCosts.__l, weights)
        self.assertTrue(np.array_equal(mock_cost.sqrt_state_weights, sqrtm(weights)))


if __name__ == '__main__':
    unittest.main()

    # def test_quadratic_nonleaf_cost_value(self):
    #     cost_item = TestCosts.__QuadraticNonleaf
    #     nonleaf_cost = cost_item.get_cost_value(TestCosts.__state, TestCosts.__control)
    #     self.assertEqual(np.shape(nonleaf_cost), ())
    #     self.assertEqual(nonleaf_cost, TestCosts.__nonleaf_cost)
    #     self.assertEqual(np.shape(cost_item.most_recent_cost_value), ())
    #     self.assertEqual(cost_item.most_recent_cost_value, nonleaf_cost)

    # def test_quadratic_leaf_cost_value(self):
    #     cost_item = TestCosts.__QuadraticLeaf
    #     leaf_cost = cost_item.get_cost_value(TestCosts.__state)
    #     self.assertEqual(np.shape(leaf_cost), ())
    #     self.assertEqual(leaf_cost, TestCosts.__leaf_cost)
    #     self.assertEqual(np.shape(cost_item.most_recent_cost_value), ())
    #     self.assertEqual(cost_item.most_recent_cost_value, leaf_cost)

    # def test_quadratic_nonleaf_cost_state_failure(self):
    #     # construct bad state
    #     state = np.ones((TestCosts.__size + 1, 1))
    #
    #     # check error raised
    #     with self.assertRaises(ValueError):
    #         TestCosts.__QuadraticNonleaf.get_cost_value(state, TestCosts.__control)

    # def test_quadratic_nonleaf_cost_control_failure(self):
    #     # construct bad control
    #     control = np.ones((TestCosts.__size + 1, 1))
    #
    #     # check error raised
    #     with self.assertRaises(ValueError):
    #         TestCosts.__QuadraticNonleaf.get_cost_value(TestCosts.__state, control)

    # def test_quadratic_leaf_cost_state_failure(self):
    #     # construct bad state
    #     state = np.ones((TestCosts.__size + 1, 1))
    #
    #     # check error raised
    #     with self.assertRaises(ValueError):
    #         TestCosts.__QuadraticLeaf.get_cost_value(state)
