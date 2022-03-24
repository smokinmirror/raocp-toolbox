import raocp.core.scenario_tree as core_tree
import raocp.core.costs as core_costs
import raocp.core.risks as core_risks


def _check_lengths(nonleaf_nodes_num, total_nodes_num, A, B, cost_item, risk_item):
    all_nodes = ["A", "B", "cost_item"]
    nonleaf_nodes = ["risk_item"]
    for name in all_nodes:
        length = len(eval(name))
        if length != total_nodes_num:
            raise ValueError('incorrect dimension in list `%s`, len(%s) = %d, number of nodes = %d'
                             % (name, name, length, total_nodes_num))
    for name in nonleaf_nodes:
        length = len(eval(name))
        if length != nonleaf_nodes_num:
            raise ValueError('incorrect dimension in list `%s`, len(%s) = %d, number of nonleaf nodes = %d'
                             % (name, name, length, nonleaf_nodes_num))


class RAOCP:
    """
    Risk-averse optimal control problem creation and storage
    """

    def __init__(self, last_nonleaf_node, last_leaf_node,
                 system_dynamics, input_dynamics,
                 cost_item, risk_item):
        """
        :param last_nonleaf_node: last node at stage N-1
        :param last_leaf_node: last node at stage N
        :param system_dynamics: list of the system dynamics (A) at each node
        :param input_dynamics: list of the input dynamics (B) at each node
        :param cost_item: list of cost class at each node
        :param risk_item: list of risk class at each nonleaf node

        Note: ambiguity sets of coherent risk measures can be expressed by conic inequalities,
                defined by a tuple (E, F, cone, b)
        Note: avoid using this constructor directly; use the builder instead
        """
        # Nodes
        self.__num_nonleaf_node = last_nonleaf_node
        self.__num_leaf_node = last_leaf_node
        # System
        self.__A = system_dynamics
        self.__B = input_dynamics
        self.__state = []
        self.__input = []
        # Cost
        self.__cost_item = cost_item
        self.__cost_value = []
        # Risk
        self.__risk_item = risk_item
        self.__risk_value = []

    # GETTERS
    @property
    def num_nonleaf_nodes(self):
        """Total number of nonleaf nodes"""
        return self.__num_nonleaf_node

    @property
    def num_nodes(self):
        """Total number of nodes"""
        return self.__num_leaf_node

    def A_at_node(self, idx):
        """
        :param idx: node index
        :return: A matrix at node idx
        """
        return self.__A[idx]

    def B_at_node(self, idx):
        """
        :param idx: node index
        :return: B matrix at node idx
        """
        return self.__B[idx]

    def cost_item_at_node(self, idx):
        """
        :param idx: node index
        :return: cost class at node idx
        """
        return self.__cost_item[idx]

    def risk_item_at_node(self, idx):
        """
        :param idx: node index
        :return: risk class at node idx
        """
        return self.__risk_item[idx]

    # SETTERS
    def update_state(self, state):
        """
        :param state: list of calculated state (x) at each node
        :return: nothing
        """
        self.__state = state

    def update_input(self, ip):
        """
        :param ip: list of calculated input (u) at each node
        :return: nothing
        """
        self.__input = ip

    def update_cost_value(self, cost_value):
        """
        :param cost_value: list of calculated cost at each node
        :return: nothing
        """
        self.__cost_value = cost_value

    def update_risk_value(self, risk_value):
        """
        :param risk_value: list of calculated risk at each node
        :return: nothing
        """
        self.__risk_value = risk_value

    def __str__(self):
        return f"RAOCP\n+ Nodes: {self.__num_leaf_node}\n" \
               f"+ {self.__cost_item[0]}\n" \
               f"+ {self.__risk_item[0]}"

    def __repr__(self):
        return f"RAOCP with {self.__num_leaf_node} nodes, root cost: {self.__cost_item[0].type()}, " \
               f"root risk: {self.__risk_item[0].type()}."


class MarkovChainRAOCPProblemBuilder:
    """
    Configuration class for easy building of RAOCP
    """
    def __init__(self, scenario_tree: core_tree.ScenarioTree):
        self.__tree = scenario_tree
        self.__A = None
        self.__B = None
        self.__cost_item = None
        self.__risk_item = None

    # SETTERS
    def with_possible_As_and_Bs(self, possible_As, possible_Bs):
        self.__A = [None]
        self.__B = [None]
        for i in range(1, self.__tree.num_nodes()):
            self.__A.append(possible_As[self.__tree.value_at_node(i)])
            self.__B.append(possible_Bs[self.__tree.value_at_node(i)])
        return self

    def with_all_cost(self, cost_type, Q, R, Pf):
        self.__cost_item = []
        if cost_type == "quadratic":
            for i in range(self.__tree.num_nodes()):
                self.__cost_item.append(core_costs.Quadratic(Q, R, Pf, i))
            return self
        else:
            raise ValueError('cost type %s not supported' % cost_type)

    def with_all_risk(self, risk_type, alpha):
        self.__risk_item = []
        if risk_type == "AVaR":
            for i in range(self.__tree.num_nonleaf_nodes()):
                self.__risk_item.append(core_risks.AVaR(alpha,
                                                        self.__tree.conditional_probabilities_of_children(i),
                                                        i))
            return self
        else:
            raise ValueError('risk type %s not supported' % risk_type)

    def create(self):
        """
        Checks lengths of each list and then creates a risk-averse optimal control problem
        """
        _check_lengths(self.__tree.num_nonleaf_nodes(), self.__tree.num_nodes(),
                       self.__A, self.__B,
                       self.__cost_item, self.__risk_item)
        problem = RAOCP(self.__tree.num_nonleaf_nodes(), self.__tree.num_nodes(),
                        self.__A, self.__B,
                        self.__cost_item, self.__risk_item)
        return problem
