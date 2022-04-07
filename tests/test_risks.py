import unittest
import raocp.core.risks as core_risks
import numpy as np


class TestRisks(unittest.TestCase):
    __AVaR = None
    __test_num_children = 10
    __num_test_repeats = 100

    @staticmethod
    def __create_test_avar():
        TestRisks.__AVaR = core_risks.AVaR(0.5, [1 / TestRisks.__test_num_children] * TestRisks.__test_num_children)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRisks.__create_test_avar()

    def test_alpha_value(self):
        test_alphas = [0, 0.5, 1]
        for i in test_alphas:
            core_risks.AVaR(i, np.zeros(2))

    def test_alpha_value_failure(self):
        test_alphas = [-0.1, 1.1]
        for i in test_alphas:
            with self.assertRaises(ValueError):
                core_risks.AVaR(i, np.zeros(2))

    def test_dimension_check_e(self):
        avar = TestRisks.__AVaR
        self.assertEqual(avar.matrix_e.shape, (2 * TestRisks.__test_num_children + 1, TestRisks.__test_num_children))

    def test_dimension_check_f(self):
        avar = TestRisks.__AVaR
        self.assertEqual(avar.matrix_f.shape, (0, TestRisks.__test_num_children))

    def test_dimension_check_cone(self):
        avar = TestRisks.__AVaR
        self.assertEqual(avar.cone.dimension, 2 * TestRisks.__test_num_children + 1)

    def test_dimension_check_b(self):
        avar = TestRisks.__AVaR
        self.assertEqual(avar.vector_b.shape, (2 * TestRisks.__test_num_children + 1, 1))


if __name__ == '__main__':
    unittest.main()
