import numpy as np

import raocp.core.scenario_tree as core_tree
import raocp.core.costs as core_costs
import raocp.core.risks as core_risks


class RAOCP:
    """
    Risk-averse optimal control problem creation and storage
    """

    def __init__(self, scenario_tree: core_tree.ScenarioTree):
        """
        :param scenario_tree: instance of ScenarioTree
        """
        self.__tree = scenario_tree
        self.__list_of_system_dynamics = [None] * self.__tree.num_nodes  # matrix A
        self.__list_of_control_dynamics = [None] * self.__tree.num_nodes  # matrix B
        self.__list_of_cost_items = [None] * self.__tree.num_nodes
        self.__list_of_risk_items = [None] * self.__tree.num_nonleaf_nodes

    # GETTERS
    @property
    def list_of_system_dynamics(self):
        return self.__list_of_system_dynamics

    @property
    def list_of_control_dynamics(self):
        return self.__list_of_control_dynamics

    @property
    def list_of_cost_items(self):
        return self.__list_of_cost_items

    @property
    def list_of_risk_items(self):
        return self.__list_of_risk_items

    # SETTERS
    def with_markovian_dynamics(self, system_dynamics, control_dynamics):
        if len(system_dynamics) != len(control_dynamics):
            raise ValueError("number of Markovian system dynamics matrices not equal to "
                             "number of Markovian control dynamics matrices")
        for i in range(len(system_dynamics)):
            if system_dynamics[i].shape != system_dynamics[0].shape:
                raise ValueError("Markovian system dynamics matrices are different shapes")
            if control_dynamics[i].shape != control_dynamics[0].shape:
                raise ValueError("Markovian control dynamics matrices are different shapes")
            if system_dynamics[i].shape[0] != control_dynamics[i].shape[0]:
                raise ValueError("Markovian dynamics matrices rows are different sizes")

        if self.__tree.tree_factory == "MarkovChain":
            for i in range(1, self.__tree.num_nodes):
                self.__list_of_system_dynamics[i] = system_dynamics[self.__tree.value_at_node(i)]
                self.__list_of_control_dynamics[i] = control_dynamics[self.__tree.value_at_node(i)]
            return self
        else:
            raise TypeError('dynamics provided as Markovian, scenario tree provided is not Markovian')

    def with_all_costs(self, cost_type, nonleaf_state_weights, control_weights, leaf_state_weights):
        if nonleaf_state_weights.shape[1] != leaf_state_weights.shape[1]:
            raise ValueError("nonleaf and leaf state cost weight matrices columns are different sizes")
        if isinstance(control_weights, np.ndarray):
            if nonleaf_state_weights.shape[0] != control_weights.shape[0]:
                raise ValueError("state cost weight matrix rows not equal to control cost weight matrix rows")

        if cost_type == "Quadratic":
            if nonleaf_state_weights.shape != leaf_state_weights.shape:
                raise ValueError("nonleaf and leaf state cost weight matrices are different shapes")
            for i in range(self.__tree.num_nodes):
                if i < self.__tree.num_nonleaf_nodes:
                    self.__list_of_cost_items[i] = core_costs.QuadraticNonleaf(nonleaf_state_weights,
                                                                               control_weights)
                else:
                    self.__list_of_cost_items[i] = core_costs.QuadraticLeaf(leaf_state_weights)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    def with_all_risks(self, risk_type, alpha):
        if risk_type == "AVaR":
            for i in range(self.__tree.num_nonleaf_nodes):
                self.__list_of_risk_items[i] = core_risks.AVaR(alpha,
                                                               self.__tree.conditional_probabilities_of_children(i))
            return self
        else:
            raise ValueError("risk type '%s' not supported" % risk_type)

    def __str__(self):
        return f"RAOCP\n+ Nodes: {self.__tree.num_nodes}\n" \
               f"+ {self.__list_of_cost_items[0]}\n" \
               f"+ {self.__list_of_risk_items[0]}"

    def __repr__(self):
        return f"RAOCP with {self.__tree.num_nodes} nodes, " \
               f"with root cost: {type(self.__list_of_cost_items[0]).__name__}, " \
               f"with root risk: {type(self.__list_of_risk_items[0]).__name__}."
