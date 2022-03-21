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
            # data = {
            #     "cost": {
            #         "type": "quadratic",
            #         "Q": np.eye(4),
            #         "R": np.eye(2)
            #     },
            #     "constraints": {
            #         "type": "polyhedral",
            #         "x_min": -np.ones(4, ),
            #         "x_max": np.ones(4, ),
            #         "u_min": -np.ones(2, ),
            #         "u_max": np.ones(2, )
            #     },
            #     "dynamics": {
            #         "type": "linear",
            #         "A": np.random.rand(4, 4),
            #         "B": np.random.rand(2, 4)
            #     },
            #     "risk": {
            #         "type": "AV@R",
            #         "alpha": 0.7,
            #         "E": 1,
            #         "F": 1,
            #         "b": None
            #     }
            # }
            tree = TestRAOCP.__tree_from_markov
            tree.set_data_at_node(5, data)
            x_5 = np.array([[1],
                           [1]])
            u_5 = np.array([[1],
                           [1]])
            TestRAOCP.__tree_from_markov = \
                rc.MarkovChainRAOCPFactory(x_5, u_5, tree.get_data_at_node(5), tree.value_at_node(5)).create()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRAOCP.__construct_tree_from_markov()
        TestRAOCP.__construct_raocp_from_markov()

    # def test_cost(self):


