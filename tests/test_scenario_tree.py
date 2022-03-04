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
        self.fail()

    def test_markov_children_of(self):
        self.fail()

    def test_markov_nodes_at_stage(self):
        self.fail()

    def test_markov_probability_of_node(self):
        self.fail()

    def test_markov_siblings_of_node(self):
        self.fail()

    def test_markov_conditional_probabilities_of_children(self):
        self.fail()
