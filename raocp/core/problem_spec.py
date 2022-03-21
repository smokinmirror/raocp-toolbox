import numpy as np
import raocp.core.scenario_tree as core_tree


def _check_lengths(nonleaf_nodes_num, total_nodes_num, A, B, cost_type, Q, R, Pf, risk_type, alpha, E, F, cone, b):
    all_nodes = ["A", "B", "cost_type", "Q", "R", "Pf"]
    nonleaf_nodes = ["risk_type", "alpha", "E", "F", "cone", "b"]
    for name in all_nodes:
        length = len(eval(name))
        if length != total_nodes_num:
            raise ValueError('incorrect dimension in list `%s`, len(%s) = %d, number of nodes = %d'
                             % (name, name, length, total_nodes_num))
    for name in nonleaf_nodes:
        length = len(eval(name))
        if length != nonleaf_nodes_num:
            raise ValueError('incorrect dimension in list `%s`, len(%s) = %d, number of nonleaf nodes = %d'
                             % (name, name, length, nonleaf_nodes_num()))


class RAOCP:
    """
    Risk-averse optimal control problem creation and storage
    """

    def __init__(self, last_nonleaf_node, last_leaf_node,
                 root_state, system_dynamics, input_dynamics,
                 cost_type, Q, R, Pf,
                 risk_type, alpha, E, F, cone, b):
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
        :param cone: list of cloco cone K for each node
        :param b: list of b vector for each node

        Note: ambiguity sets of coherent risk measures can be expressed by conic inequalities,
                defined by the tuple (E, F, cone, b)
        Note: avoid using this constructor directly; use a factory instead
        """
        # Nodes
        self.__last_nonleaf_node = last_nonleaf_node
        self.__last_leaf_node = last_leaf_node
        # System
        self.__A = system_dynamics
        self.__B = input_dynamics
        self.__state = [root_state]
        self.__input = []
        # Cost
        self.__cost_type = cost_type
        self.__Q = Q
        self.__R = R
        self.__Pf = Pf
        self.__cost = []
        # Risk
        self.__risk_type = risk_type
        self.__alpha = alpha
        self.__E = E
        self.__F = F
        self.__cone = cone
        self.__b = b
        self.__risk = []

    # GETTERS
    @property
    def num_nonleaf_nodes(self):
        """Total number of nonleaf nodes"""
        return self.__last_nonleaf_node

    @property
    def num_nodes(self):
        """Total number of nodes"""
        return self.__last_leaf_node

    def A_at_node(self, idx):
        """
        :param idx: node index
        :return: A matrix at node idx
        """
        return self.__A[idx]

    # B_at_node() ...

    # SETTERS
    def update_state_at_node(self, idx, state):
        """
        :param idx: node index
        :param state: calculated state at node idx or new root node state
        :return: nothing
        """
        self.__state[idx] = state

    # update input at idx

    def update_cost_at_node(self, idx, cost):
        """
        :param idx: node index
        :param cost: calculated cost at node idx
        :return: nothing
        """
        self.__cost[idx] = cost

    # update risk at idx

    def __str__(self):
        return f"RAOCP\n+ Nodes: {self.__last_leaf_node}"

    def __repr__(self):
        return f"RAOCP with {self.__last_leaf_node} nodes"


class MarkovChainRAOCPProblemBuilder:
    """
    Configuration class for easy building of RAOCP
    """
    def __init__(self, scenario_tree: core_tree.ScenarioTree):
        self.__tree = scenario_tree
        self.__root_state = None
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
        self.__cone = None
        self.__b = None

    # SETTERS
    def with_root_state(self, root_state):
        self.__root_state = root_state
        return self

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

    def with_all_cone(self, cone):
        self.__cone = []
        for i in range(self.__tree.num_nonleaf_nodes()):
            self.__cone.append(cone)
        return self

    def with_all_b(self, b):
        self.__b = []
        for i in range(self.__tree.num_nonleaf_nodes()):
            self.__b.append(b)
        return self

    def create(self):
        """
        Checks lengths of each list and then creates a risk-averse optimal control problem
        """
        _check_lengths(self.__tree.num_nonleaf_nodes(), self.__tree.num_nodes(),
                       self.__A, self.__B,
                       self.__cost_type, self.__Q, self.__R, self.__Pf,
                       self.__risk_type, self.__alpha, self.__E, self.__F, self.__cone, self.__b)
        problem = RAOCP(self.__tree.num_nonleaf_nodes(), self.__tree.num_nodes(),
                        self.__root_state, self.__A, self.__B,
                        self.__cost_type, self.__Q, self.__R, self.__Pf,
                        self.__risk_type, self.__alpha, self.__E, self.__F, self.__cone, self.__b)
        return problem


class RAOCPFactory:
    """
    Factory class to construct risk-averse optimal control problem

    Only for when the user wishes to provide all the lists in their entirety
    """
    def __init__(self, scenario_tree: core_tree.ScenarioTree, root_state,
                 A=None, B=None,
                 cost_type=None, Q=None, R=None, Pf=None,
                 risk_type=None, alpha=None, E=None, F=None, cone=None, b=None):
        """
        :param scenario_tree: instance of ScenarioTree class
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
        :param cone: list of cloco cone K for each node
        :param b: list of b vector for each node
        """
        self.__tree = scenario_tree
        self.__root_state = root_state
        self.__A = A
        self.__B = B
        self.__cost_type = cost_type
        self.__Q = Q
        self.__R = R
        self.__Pf = Pf
        self.__risk_type = risk_type
        self.__alpha = alpha
        self.__E = E
        self.__F = F
        self.__cone = cone
        self.__b = b

    def create(self):
        """
        Checks lengths of each list and then creates a risk-averse optimal control problem
        """
        _check_lengths(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes,
                       self.__A, self.__B,
                       self.__cost_type, self.__Q, self.__R, self.__Pf,
                       self.__risk_type, self.__alpha, self.__E, self.__F, self.__cone, self.__b)
        problem = RAOCP(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes,
                        self.__root_state, self.__A, self.__B,
                        self.__cost_type, self.__Q, self.__R, self.__Pf,
                        self.__risk_type, self.__alpha, self.__E, self.__F, self.__cone, self.__b)
        return problem
