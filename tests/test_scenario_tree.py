from unittest import TestCase
import raocp.core as rc
import numpy as np


class TestScenarioTree(TestCase):

    __tree_from_markov = None
    __tree_from_iid = None

    @staticmethod
    def __construct_tree_from_markov():
        if TestScenarioTree.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1], [0.4, 0.6, 0], [0, 0.3, 0.7]])
            v = np.array([0.5, 0.5, 0])
            (N, tau) = (4, 3)
            TestScenarioTree.__tree_from_markov = \
                rc.ScenarioTree.from_markov_chain(p, v, N, tau)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestScenarioTree.__construct_tree_from_markov()

    def test_markov_num_nodes(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(32, tree.num_nodes())

    def test_markov_ancestor_of(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(0, tree.ancestor_of(1))
        self.assertEqual(0, tree.ancestor_of(2))
        self.assertEqual(1, tree.ancestor_of(3))
        self.assertEqual(1, tree.ancestor_of(4))
        self.assertEqual(1, tree.ancestor_of(5))
        self.assertEqual(2, tree.ancestor_of(6))
        self.assertEqual(2, tree.ancestor_of(7))
        self.assertEqual(5, tree.ancestor_of(13))
        self.assertEqual(8, tree.ancestor_of(20))
        self.assertEqual(15, tree.ancestor_of(27))
        self.assertEqual(19, tree.ancestor_of(31))

    def test_markov_children_of(self):
        tree = TestScenarioTree.__tree_from_markov
        self.fail()

    def test_markov_nodes_at_stage(self):
        tree = TestScenarioTree.__tree_from_markov
        self.fail()

    def test_markov_probability_of_node(self):
        tol = 1e-10
        tree = TestScenarioTree.__tree_from_markov
        self.assertAlmostEqual(1, tree.probability_of_node(0), delta=tol)
        self.assertAlmostEqual(0.5, tree.probability_of_node(1), delta=tol)
        self.assertAlmostEqual(0.5, tree.probability_of_node(2), delta=tol)
        self.assertAlmostEqual(0.05, tree.probability_of_node(3), delta=tol)
        self.assertAlmostEqual(0.4, tree.probability_of_node(4), delta=tol)
        self.assertAlmostEqual(0.05, tree.probability_of_node(5), delta=tol)
        self.assertAlmostEqual(0.2, tree.probability_of_node(6), delta=tol)
        self.assertAlmostEqual(0.3, tree.probability_of_node(7), delta=tol)
        self.assertAlmostEqual(0.005, tree.probability_of_node(8), delta=tol)

    def test_markov_siblings_of_node(self):
        self.fail()

    def test_markov_conditional_probabilities_of_children(self):
        self.fail()
