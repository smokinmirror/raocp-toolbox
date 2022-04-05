import raocp.core.scenario_tree as core_tree
import raocp.core.costs as core_costs
import raocp.core.risks as core_risks


def _check_lengths(num_nonleaf_nodes, num_nodes, A, B, cost_item, risk_item):
    all_nodes = ["A", "B", "cost_item"]
    nonleaf_nodes = ["risk_item"]
    for name in all_nodes:
        length = len(eval(name))
        if length != num_nodes:
            raise ValueError('incorrect dimension in list `%s`, len(%s) = %d, number of nodes = %d'
                             % (name, name, length, num_nodes))
    for name in nonleaf_nodes:
        length = len(eval(name))
        if length != num_nonleaf_nodes:
            raise ValueError('incorrect dimension in list `%s`, len(%s) = %d, number of nonleaf nodes = %d'
                             % (name, name, length, num_nonleaf_nodes))


class RAOCP:
    """
    Risk-averse optimal control problem creation and storage
    """

    def __init__(self, scenario_tree: core_tree.ScenarioTree):
        """
        :param scenario_tree: instance of ScenarioTree
        """
        self.__tree = scenario_tree
        self.__list_of_system_dynamics = [None] * self.__tree.num_nodes()  # matrix A
        self.__list_of_input_dynamics = [None] * self.__tree.num_nodes()  # matrix B
        self.__list_of_cost_items = [None] * self.__tree.num_nodes()
        self.__list_of_risk_items = [None] * self.__tree.num_nonleaf_nodes()

    # GETTERS
    @property
    def list_of_system_dynamics(self):
        return self.__list_of_system_dynamics

    @property
    def list_of_input_dynamics(self):
        return self.__list_of_input_dynamics

    @property
    def list_of_cost_items(self):
        return self.__list_of_cost_items

    @property
    def list_of_risk_items(self):
        return self.__list_of_risk_items

    # SETTERS
    def with_markovian_dynamics(self, system_dynamics, input_dynamics):
        if self.__tree.tree_factory == "MarkovChain":
            for i in range(1, self.__tree.num_nodes):
                self.__list_of_system_dynamics[i] = system_dynamics[self.__tree.value_at_node(i)]
                self.__list_of_input_dynamics[i] = input_dynamics[self.__tree.value_at_node(i)]
            return self
        else:
            raise TypeError('dynamics are Markovian, but scenario tree is not')

    def with_all_costs(self, cost_type, nonleaf_state_weights, input_weights, leaf_state_weights):
        if cost_type == "Quadratic":
            for i in range(self.__tree.num_nodes):
                if i < self.__tree.num_nonleaf_nodes:
                    self.__list_of_cost_items[i] = core_costs.QuadraticNonleaf(Q, R, i)
                else:
                    self.__list_of_cost_items[i] = core_costs.QuadraticLeaf(Pf, i)
            return self
        else:
            raise ValueError('cost type %s not supported' % cost_type)

    def with_all_risks(self, risk_type, alpha):
        self.__risk_item = []
        if risk_type == "AVaR":
            for i in range(self.__tree.num_nonleaf_nodes()):
                self.__risk_item.append(core_risks.AVaR(alpha, self.__tree.conditional_probabilities_of_children(i)))
            return self
        else:
            raise ValueError('risk type %s not supported' % risk_type)

    def __str__(self):
        return f"RAOCP\n+ Nodes: {self.__tree.num_nodes()}\n" \
               f"+ {self.__list_of_cost_items[0]}\n" \
               f"+ {self.__list_of_risk_items[0]}"

    def __repr__(self):
        return f"RAOCP with {self.__tree.num_nodes()} nodes, " \
               f"with root cost: {self.__list_of_cost_items[0].type()}, " \
               f"with root risk: {self.__list_of_risk_items[0].type()}."



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
