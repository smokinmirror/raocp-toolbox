import unittest
import numpy as np
import raocp.core.risks as core_risks


class TestRisks(unittest.TestCase):
    __AVaR = None
    __test_num_children = 10
    __num_test_repeats = 100

    @staticmethod
    def _create_test_avar():
        TestRisks.__AVaR = core_risks.AVaR(0.5)
        probs = np.array([1 / TestRisks.__test_num_children] * TestRisks.__test_num_children).reshape(-1, 1)
        TestRisks.__AVaR.probs = probs

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRisks._create_test_avar()

    def test_is_risk(self):
        self.assertTrue(core_risks.AVaR.is_risk)

    def test_alpha_value(self):
        test_alphas = [0, 0.5, 1]
        for i in test_alphas:
            core_risks.AVaR(i)

    def test_alpha_value_failure(self):
        test_alphas = [-0.1, 1.1]
        for i in test_alphas:
            with self.assertRaises(ValueError):
                core_risks.AVaR(i)

    def test_probs(self):
        num = 10
        probs = np.random.sample(num).reshape(-1, 1)
        risk = core_risks.AVaR(0)
        risk.probs = probs
        self.assertTrue(np.array_equal(risk.probs, probs))
        self.assertTrue(risk._AVaR__num_children == len(probs))

    def test_dimension_check_e(self):
        avar = TestRisks.__AVaR
        self.assertEqual(avar.matrix_e.shape, (2 * TestRisks.__test_num_children + 1, TestRisks.__test_num_children))

    def test_dimension_check_f(self):
        avar = TestRisks.__AVaR
        self.assertEqual(avar.matrix_f.shape, (2 * TestRisks.__test_num_children + 1, 0))

    def test_dimension_check_cone(self):
        avar = TestRisks.__AVaR
        self.assertEqual(avar.cone.dimension, 2 * TestRisks.__test_num_children + 1)

    def test_dimension_check_b(self):
        avar = TestRisks.__AVaR
        self.assertEqual(avar.vector_b.shape, (2 * TestRisks.__test_num_children + 1, 1))


if __name__ == '__main__':
    unittest.main()
