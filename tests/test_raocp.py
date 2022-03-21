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
            tree = TestRAOCP.__tree_from_markov
            x0 = np.array([[1], [1]])

            Aw1 = np.eye(2)
            Aw2 = np.eye(2)
            Aw3 = np.eye(2)
            As = [Aw1, Aw2, Aw3]

            Bw1 = np.eye(2)
            Bw2 = np.eye(2)
            Bw3 = np.eye(2)
            Bs = [Bw1, Bw2, Bw3]

            cost_type = "quadratic"
            (Q, R, Pf) = (np.eye(2), np.eye(2), np.eye(2))

            (risk_type, alpha) = ("AVAR", 0.5)
            (E, F, Kone, b) = (np.eye(2),  # p x n matrix
                               np.eye(2),  # p x r matrix
                               "Rn+",
                               np.ones((2, 1)))  # p vector

            TestRAOCP.__raocp_config = rc.RAOCPconfig(scenario_tree=tree) \
                .with_possible_As_and_Bs(As, Bs) \
                .with_all_cost_type(cost_type).with_all_Q(Q).with_all_R(R).with_all_Pf(Pf) \
                .with_all_risk_type(risk_type).with_all_alpha(alpha).with_all_E(E).with_all_F(F).with_all_Kone(
                Kone).with_all_b(b)
            TestRAOCP.__raocp_from_markov = rc.RAOCPfactory(problem_config=TestRAOCP.__raocp_config,
                                                            root_state=x0).create()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRAOCP.__construct_tree_from_markov()
        TestRAOCP.__construct_raocp_from_markov()


