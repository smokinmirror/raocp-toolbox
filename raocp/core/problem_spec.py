import numpy as np
import raocp.core.scenario_tree as core_tree


class RAOCP:
    """
    Risk-averse optimal control problem creation
    """

    def __init__(self, last_nonleaf_node, last_leaf_node,
                 root_state, system_dynamics, input_dynamics,
                 cost_type, Q, R, Pf,
                 risk_type, alpha, E, F, Kone, b):
        """
        :param last_nonleaf_node: last node at stage N-1
        :param last_leaf_node: last node at stage N
        :param root_state: system state at node 0
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
        :param Kone: list of cloco cone K for each node
        :param b: list of b vector for each node

        Note: ambiguity sets of coherent risk measures can be expressed by conic inequalities,
                defined by the tuple (E, F, Kone, b)
        Note: avoid using this constructor directly; use a factory instead
        """
        # Nodes
        self.__last_nonleaf_node = last_nonleaf_node
        self.__last_leaf_node = last_leaf_node
        # System
        self.__root_state = root_state
        self.__system_dynamics = system_dynamics
        self.__input_dynamics = input_dynamics
        self.__state = None
        self.__input = None
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
        self.__A = None
        self.__B = None
        self.__cost_type = None
        self.__Q = None
        self.__R = None
        self.__Pf = None
        self.__risk_type = None
        self.__alpha = None
        self.__E = None
        self.__F = None
        self.__Kone = None
        self.__b = None

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
    def A(self):
        """List of A at each node"""
        return self.__A

    @property
    def B(self):
        """List of B at each node"""
        return self.__B

    @property
    def cost_type(self):
        """List of cost type at each node"""
        return self.__cost_type

    @property
    def Q(self):
        """List of Q at each node"""
        return self.__Q

    @property
    def R(self):
        """List of R at each node"""
        return self.__R

    @property
    def Pf(self):
        """List of Pf at each node"""
        return self.__Pf

    @property
    def risk_type(self):
        """List of risk type at each node"""
        return self.__risk_type

    @property
    def alpha(self):
        """List of alpha at each node"""
        return self.__alpha

    @property
    def E(self):
        """List of E at each node"""
        return self.__E

    @property
    def F(self):
        """List of F at each node"""
        return self.__F

    @property
    def Kone(self):
        """List of Kone at each node"""
        return self.__Kone

    @property
    def b(self):
        """List of b at each node"""
        return self.__b

    # SETTERS
    def with_possible_As_and_Bs(self, possible_As, possible_Bs):
        self.__A = [None]
        self.__B = [None]
        for i in range(1, self.__tree.num_nodes()):
            self.__A.append(possible_As[self.__tree.value_at_node(i)])
            self.__B.append(possible_Bs[self.__tree.value_at_node(i)])
        return self

    def with_all_cost_type(self, cost_type):
        self.__cost_type = []
        for i in range(self.__tree.num_nodes()):
            self.__cost_type.append(cost_type)
        return self

    def with_all_Q(self, Q):
        self.__Q = []
        for i in range(self.__tree.num_nodes()):
            self.__Q.append(Q)
        return self

    def with_all_R(self, R):
        self.__R = []
        for i in range(self.__tree.num_nodes()):
            self.__R.append(R)
        return self

    def with_all_Pf(self, Pf):
        self.__Pf = []
        for i in range(self.__tree.num_nodes()):
            self.__Pf.append(Pf)
        return self

    def with_all_risk_type(self, risk_type):
        self.__risk_type = []
        for i in range(self.__tree.num_nonleaf_nodes()):
            self.__risk_type.append(risk_type)
        return self

    def with_all_alpha(self, alpha):
        self.__alpha = []
        for i in range(self.__tree.num_nonleaf_nodes()):
            self.__alpha.append(alpha)
        return self

    def with_all_E(self, E):
        self.__E = []
        for i in range(self.__tree.num_nonleaf_nodes()):
            self.__E.append(E)
        return self

    def with_all_F(self, F):
        self.__F = []
        for i in range(self.__tree.num_nonleaf_nodes()):
            self.__F.append(F)
        return self

    def with_all_Kone(self, Kone):
        self.__Kone = []
        for i in range(self.__tree.num_nonleaf_nodes()):
            self.__Kone.append(Kone)
        return self

    def with_all_b(self, b):
        self.__b = []
        for i in range(self.__tree.num_nonleaf_nodes()):
            self.__b.append(b)
        return self


def _check_lengths(num_nonleaf_nodes, num_nodes, A, B, cost_type, Q, R, Pf, risk_type, alpha, E, F, Kone, b):
    all_nodes = ["A", "B", "cost_type", "Q", "R", "Pf"]
    nonleaf_nodes = ["risk_type", "alpha", "E", "F", "Kone", "b"]
    for name in all_nodes:
        if len(eval(name)) != num_nodes:
            raise ValueError('incorrect dimension in list `%s`, len(%s) = %d, number of nodes = %d'
                             % (name, name, len(eval(name)), num_nodes))
    for name in nonleaf_nodes:
        if len(eval(name)) != num_nonleaf_nodes:
            raise ValueError('incorrect dimension in list `%s`, len(%s) = %d, number of nonleaf nodes = %d'
                             % (name, name, len(eval(name)), num_nonleaf_nodes))


class RAOCPfactory:
    """
    Factory class to construct risk-averse optimal control problem.

    Problem lists are either generated using the `RAOCPconfig` class or supplied to this factory. Lists
    supplied to the factory take priority over `RAOCPconfig` generated lists.
    """
    def __init__(self, problem_config: RAOCPconfig, root_state,
                 A=None, B=None,
                 cost_type=None, Q=None, R=None, Pf=None,
                 risk_type=None, alpha=None, E=None, F=None, Kone=None, b=None):
        """
        :param problem_config: instance of RAOCPconfig class
        :param root_state: state at root node
        :param A: list of possible system dynamics matrices
        :param B: list of possible input dynamics matrices
        :param cost_type: list of the cost type (quadratic, ...) at each node
        :param Q: list of the Q matrix used for computing nonleaf node cost
        :param R: list of the R matrix used for computing nonleaf node cost
        :param Pf: list of the P_f matrix used for computing leaf node cost
        :param risk_type: list of the risk type (AVAR, EVAR, conic, ...) at each node
        :param alpha: list of risk parameter at each node
        :param E: list of E matrix for each node
        :param F: list of F matrix for each node
        :param Kone: list of cloco cone K for each node
        :param b: list of b vector for each node
        """
        self.__config = problem_config
        self.__root_state = root_state
        self.__A = self.__config.A if A is None else A
        self.__B = self.__config.B if B is None else B
        self.__cost_type = self.__config.cost_type if cost_type is None else cost_type
        self.__Q = self.__config.Q if Q is None else Q
        self.__R = self.__config.R if R is None else R
        self.__Pf = self.__config.Pf if Pf is None else Pf
        self.__risk_type = self.__config.risk_type if risk_type is None else risk_type
        self.__alpha = self.__config.alpha if alpha is None else alpha
        self.__E = self.__config.E if E is None else E
        self.__F = self.__config.F if F is None else F
        self.__Kone = self.__config.Kone if Kone is None else Kone
        self.__b = self.__config.b if b is None else b

    def create(self):
        """
        Creates a risk-averse optimal control problem from the given Markov chain in the ScenarioTree instance
        """
        _check_lengths(self.__config.num_nonleaf_nodes, self.__config.num_nodes,
                       self.__A, self.__B,
                       self.__cost_type, self.__Q, self.__R, self.__Pf,
                       self.__risk_type, self.__alpha, self.__E, self.__F, self.__Kone, self.__b)
        problem = RAOCP(self.__config.num_nonleaf_nodes, self.__config.num_nodes,
                        self.__root_state, self.__A, self.__B,
                        self.__cost_type, self.__Q, self.__R, self.__Pf,
                        self.__risk_type, self.__alpha, self.__E, self.__F, self.__Kone, self.__b)
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
