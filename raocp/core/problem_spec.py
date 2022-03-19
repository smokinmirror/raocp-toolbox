import numpy as np
import raocp.core.scenario_tree as core_tree


class RAOCP:
    """
    Risk-averse optimal control problem creation
    """

    def __init__(self, last_nonleaf_node, last_leaf_node,
                 state, ip, system_dynamics, input_dynamics,
                 cost_type, Q, R, Pf,
                 risk_type, alpha, E, F, Kone, b):
        """
        :param last_nonleaf_node: last node at stage N-1
        :param last_leaf_node: last node at stage N
        :param state: list of the state (x) at each node
        :param ip: list of the input (u) at each node
        :param system_dynamics: list of the system dynamics (A) at each node
        :param input_dynamics: list of the input dynamics (B) at each node
        :param cost_type: list of the cost type (quadratic, ...) at each node
        :param Q: list of the Q matrix used for computing nonleaf node cost
        :param R: list of the R matrix used for computing nonleaf node cost
        :param Pf: list of the P_f matrix used for computing leaf node cost
        :param risk_type: list of the risk type (AVAR, EVAR, conic, ...) at each node
        :param alpha: list of risk parameter at each node
        :param E: list of E matrix for each node
        :param F: list of F matrix for each node
        :param Kone: list of Kone for each node
        :param b: list of b vector for each node

        Note: ambiguity sets of coherent risk measures can be expressed by conic inequalities,
                defined by the tuple (E, F, Kone, b)
        Note: avoid using this constructor directly; use a factory instead
        """
        # Nodes
        self.__last_nonleaf_node = last_nonleaf_node
        self.__last_leaf_node = last_leaf_node
        # System
        self.__state = state
        self.__input = ip
        self.__system_dynamics = system_dynamics
        self.__input_dynamics = input_dynamics
        # Cost
        self.__cost_type = cost_type
        self.__Q = Q
        self.__R = R
        self.__Pf = Pf
        self.__cost = None
        # Risk
        self.__risk_type = risk_type
        self.__alpha = alpha
        self.__E = E
        self.__F = F
        self.__Kone = Kone
        self.__b = b
        self.__risk = None


class RAOCPconfig:
    """
    Configuration class for easy building of RAOCPFactory using ScenarioTree instance
    """
    def __init__(self, scenario_tree: core_tree.ScenarioTree):
        self.__tree = scenario_tree
        self.__As = None
        self.__Bs = None
        self.__cost_type = None
        self.__risk_type = None

    # GETTERS
    @property
    def num_nonleaf_nodes(self):
        """Total number of nonleaf nodes"""
        return self.__tree.num_nonleaf_nodes()

    @property
    def num_nodes(self):
        """Total number of nodes"""
        return self.__tree.num_nodes()

    @property
    def As(self):
        """List of A at each node"""
        return self.__As

    @property
    def Bs(self):
        """List of B at each node"""
        return self.__Bs

    @property
    def cost_type(self):
        """List of cost type at each node"""
        return self.__cost_type

    @property
    def risk_type(self):
        """List of risk type at each node"""
        return self.__risk_type

    # SETTERS
    def with_As(self, As):
        # allocate each node correct A
        return self

    def with_Bs(self, Bs):
        # allocate each node correct B
        return self

    def with_all_cost_type(self, cost_type):
        self.__cost_type = []
        for i in range(self.__tree.num_nodes()):
            self.__cost_type.append(cost_type)
        return self

    def with_all_risk_type(self, risk_type):
        self.__risk_type = []
        for i in range(self.__tree.num_nodes()):
            self.__risk_type.append(risk_type)
        return self


class MarkovChainRAOCPFactory:
    """
    Factory class to construct risk-averse optimal control problem from stopped Markov chains
    """
    def __init__(self, problem_config: RAOCPconfig, root_state, As=None, Bs=None):
        """
        :param problem_config: instance of RAOCPconfig class
        :param root_state: state at root node
        :param As: list of possible system dynamics matrices
        :param Bs: list of possible input dynamics matrices
        """
        self.__config = problem_config
        self.__root_state = root_state
        self.__As = self.__config.As if As is None else As
        self.__Bs = self.__config.Bs if Bs is None else Bs

    def create(self):
        """
        Creates a risk-averse optimal control problem from the given Markov chain in the ScenarioTree instance
        """
        # all the following to be deleted once builder function made
        state = [np.array([[1],
                           [1]])]
        ip = [np.array([[1],
                        [1]])]
        system_dynamics = [np.array([[1, 1],
                                     [1, 1]])]
        input_dynamics = [np.array([[1, 1],
                                    [1, 1]])]
        Q = [np.array([[1, 1],
                       [1, 1]])]
        R = [np.array([[1, 1],
                       [1, 1]])]
        Pf = [np.array([[1, 1],
                        [1, 1]])]
        alpha = [0.5]
        E = [np.array([[1, 1],
                       [1, 1]])]
        F = [np.array([[1, 1],
                       [1, 1]])]
        Kone = [0]
        b = [np.array([[1],
                       [1]])]

        problem = RAOCP(self.__config.num_nonleaf_nodes, self.__config.num_nodes,
                        state, ip, system_dynamics, input_dynamics,
                        self.__config.cost_type, Q, R, Pf,
                        self.__config.risk_type, alpha, E, F, Kone, b)
        return problem

    # def __make_cost(self):
    #     """
    #     :return: cost
    #     """
    #     if self.__data["cost"]["type"] == "quadratic":
    #         cost = self.__x_current.T @ self.__data["cost"]["Q"] @ self.__x_current \
    #                + self.__u_current.T @ self.__data["cost"]["R"] @ self.__u_current
    #     else:
    #         raise ValueError("cost type is not quadratic!")
    #     return cost
    #
    # def __make_linear_func(self):
    #     """
    #     :return: linear result
    #     """
    #     if self.__data["dynamics"]["type"] == "affine":
    #         x_next = self.__data["dynamics"]["A"] @ self.__x_current + \
    #                  self.__data["dynamics"]["B"] @ self.__u_current + self.__w_value
    #     elif self.__data["dynamics"]["type"] == "linear":
    #         x_next = self.__data["dynamics"]["A"] @ self.__x_current + self.__w_value
    #     else:
    #         raise ValueError("dynamics type is not affine or linear!")
    #     return x_next
    #
    # def __make_risk_value(self):
    #     """
    #     :return: risk results
    #     """
    #     if self.__data["risk"]["type"] == "AV@R":
    #         raise NotImplementedError()
    #     else:
    #         raise ValueError("risk type is not AV@R!")
