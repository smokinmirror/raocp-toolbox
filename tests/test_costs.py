import unittest
import raocp.core.costs as core_costs
import numpy as np


class TestCosts(unittest.TestCase):
    __QuadraticNonleaf = None
    __QuadraticLeaf = None
    __state = np.array([[-2], [2], [4]])
    __control = np.array([[5], [-5], [2]])
    __size = __state.size
    __multipliers = [2, 3, 4]
    __nonleaf_cost = 210
    __leaf_cost = 96

    @staticmethod
    def __construct_classes():
        nonleaf_state_weights = TestCosts.__multipliers[0] * np.eye(TestCosts.__size)
        control_weights = TestCosts.__multipliers[1] * np.eye(TestCosts.__size)
        leaf_state_weights = TestCosts.__multipliers[2] * np.eye(TestCosts.__size)
        TestCosts.__QuadraticNonleaf = core_costs.QuadraticNonleaf(nonleaf_state_weights, control_weights)
        TestCosts.__QuadraticLeaf = core_costs.QuadraticLeaf(leaf_state_weights)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestCosts.__construct_classes()

    def test_quadratic_nonleaf_cost_value(self):
        cost_item = TestCosts.__QuadraticNonleaf
        nonleaf_cost = cost_item.get_cost_value(TestCosts.__state, TestCosts.__control)
        self.assertEqual(np.shape(nonleaf_cost), ())
        self.assertEqual(nonleaf_cost, TestCosts.__nonleaf_cost)
        self.assertEqual(np.shape(cost_item.most_recent_cost_value), ())
        self.assertEqual(cost_item.most_recent_cost_value, nonleaf_cost)

    def test_quadratic_leaf_cost_value(self):
        cost_item = TestCosts.__QuadraticLeaf
        leaf_cost = cost_item.get_cost_value(TestCosts.__state)
        self.assertEqual(np.shape(leaf_cost), ())
        self.assertEqual(leaf_cost, TestCosts.__leaf_cost)
        self.assertEqual(np.shape(cost_item.most_recent_cost_value), ())
        self.assertEqual(cost_item.most_recent_cost_value, leaf_cost)

    def test_quadratic_nonleaf_cost_state_failure(self):
        # construct bad state
        state = np.ones((TestCosts.__size + 1, 1))

        # check error raised
        with self.assertRaises(ValueError):
            TestCosts.__QuadraticNonleaf.get_cost_value(state, TestCosts.__control)

    def test_quadratic_nonleaf_cost_control_failure(self):
        # construct bad control
        control = np.ones((TestCosts.__size + 1, 1))

        # check error raised
        with self.assertRaises(ValueError):
            TestCosts.__QuadraticNonleaf.get_cost_value(TestCosts.__state, control)

    def test_quadratic_leaf_cost_state_failure(self):
        # construct bad state
        state = np.ones((TestCosts.__size + 1, 1))

        # check error raised
        with self.assertRaises(ValueError):
            TestCosts.__QuadraticLeaf.get_cost_value(state)
