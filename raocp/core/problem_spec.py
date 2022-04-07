import numpy as np

import raocp.core.scenario_tree as core_tree
import raocp.core.dynamics as core_dynamics
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
        self.__list_of_dynamics = [None] * self.__tree.num_nodes
        self.__list_of_costs = [None] * self.__tree.num_nodes
        self.__list_of_risks = [None] * self.__tree.num_nonleaf_nodes

    # GETTERS
    @property
    def list_of_dynamics(self):
        return self.__list_of_dynamics

    @property
    def list_of_costs(self):
        return self.__list_of_costs

    @property
    def list_of_risks(self):
        return self.__list_of_risks

    # SETTERS
    def with_markovian_dynamics(self, ordered_list_of_state_dynamics, ordered_list_of_control_dynamics):
        # check same number of items in both lists
        if len(ordered_list_of_state_dynamics) != len(ordered_list_of_control_dynamics):
            raise ValueError("number of Markovian state dynamics matrices not equal to "
                             "number of Markovian control dynamics matrices")
        for i in range(len(ordered_list_of_state_dynamics)):
            # check all state dynamics have same shape
            if ordered_list_of_state_dynamics[i].shape != ordered_list_of_state_dynamics[0].shape:
                raise ValueError("Markovian state dynamics matrices are different shapes")
            # check if all control dynamics have same shape
            if ordered_list_of_control_dynamics[i].shape != ordered_list_of_control_dynamics[0].shape:
                raise ValueError("Markovian control dynamics matrices are different shapes")

        # check that scenario tree provided has MarkovChain type
        if self.__tree.tree_factory == "MarkovChain":
            for i in range(1, self.__tree.num_nodes):
                self.__list_of_dynamics[i] = core_dynamics.Dynamics(
                    ordered_list_of_state_dynamics[self.__tree.value_at_node(i)],
                    ordered_list_of_control_dynamics[self.__tree.value_at_node(i)])
            return self
        else:
            raise TypeError('dynamics provided as Markovian, scenario tree provided is not Markovian')

    def with_all_nonleaf_costs(self, cost_type, nonleaf_state_weights, control_weights):
        if cost_type == "Quadratic":
            for i in range(self.__tree.num_nonleaf_nodes):
                self.__list_of_costs[i] = core_costs.QuadraticNonleaf(nonleaf_state_weights, control_weights)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    def with_all_leaf_costs(self, cost_type, leaf_state_weights):
        if cost_type == "Quadratic":
            for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
                self.__list_of_costs[i] = core_costs.QuadraticLeaf(leaf_state_weights)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    def with_all_risks(self, risk_type, alpha):
        if risk_type == "AVaR":
            for i in range(self.__tree.num_nonleaf_nodes):
                self.__list_of_risks[i] = core_risks.AVaR(alpha, self.__tree.conditional_probabilities_of_children(i))
            return self
        else:
            raise ValueError("risk type '%s' not supported" % risk_type)

    def __str__(self):
        return f"RAOCP\n+ Nodes: {self.__tree.num_nodes}\n" \
               f"+ {self.__list_of_costs[0]}\n" \
               f"+ {self.__list_of_risks[0]}"

    def __repr__(self):
        return f"RAOCP with {self.__tree.num_nodes} nodes, " \
               f"with root cost: {type(self.__list_of_costs[0]).__name__}, " \
               f"with root risk: {type(self.__list_of_risks[0]).__name__}."
