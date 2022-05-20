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
        self.__list_of_nonleaf_costs = [None] * self.__tree.num_nodes
        self.__list_of_leaf_costs = [None] * self.__tree.num_nodes
        self.__list_of_risks = [None] * self.__tree.num_nonleaf_nodes

    # GETTERS
    @property
    def tree(self):
        return self.__tree

    @property
    def list_of_dynamics(self):
        return self.__list_of_dynamics

    @property
    def list_of_nonleaf_costs(self):
        return self.__list_of_nonleaf_costs

    @property
    def list_of_leaf_costs(self):
        return self.__list_of_leaf_costs

    @property
    def list_of_risks(self):
        return self.__list_of_risks

    def state_dynamics_at_node(self, idx):
        return self.list_of_dynamics[idx].state_dynamics

    def control_dynamics_at_node(self, idx):
        return self.list_of_dynamics[idx].control_dynamics

    def nonleaf_cost_at_node(self, idx):
        return self.list_of_nonleaf_costs[idx]

    def leaf_cost_at_node(self, idx):
        return self.list_of_leaf_costs[idx]

    def risk_at_node(self, idx):
        return self.list_of_risks[idx]

    # Dynamics ---------------------------------------------------------------------------------------------------------

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

        # check that scenario tree provided is Markovian
        if self.__tree.is_markovian:
            for i in range(1, self.__tree.num_nodes):
                self.__list_of_dynamics[i] = core_dynamics.Dynamics(
                    ordered_list_of_state_dynamics[self.__tree.value_at_node(i)],
                    ordered_list_of_control_dynamics[self.__tree.value_at_node(i)])
            return self
        else:
            raise TypeError("dynamics provided as Markovian, scenario tree provided is not Markovian")

    # Costs ------------------------------------------------------------------------------------------------------------

    def with_markovian_costs(self, ordered_list_of_cost_types, ordered_list_of_state_weights,
                             ordered_list_of_control_weights):
        # check same number of items in lists
        if sum((len(ordered_list_of_cost_types),
                len(ordered_list_of_state_weights),
                len(ordered_list_of_control_weights))) / 3 != len(ordered_list_of_cost_types):
            raise ValueError("Markovian costs lists provided are not of equal sizes")
        # check that scenario tree is Markovian
        if self.__tree.is_markovian:
            for i in range(1, self.__tree.num_nodes):
                if ordered_list_of_cost_types[self.__tree.value_at_node(i)] == "Quadratic":
                    self.__list_of_nonleaf_costs[i] = core_costs.QuadraticNonleaf(
                        ordered_list_of_state_weights[self.__tree.value_at_node(i)],
                        ordered_list_of_control_weights[self.__tree.value_at_node(i)])
                else:
                    raise ValueError("cost type '%s' not supported" % ordered_list_of_cost_types[i])
            return self
        else:
            raise TypeError("costs provided as Markovian, scenario tree provided is not Markovian")

    def with_all_nonleaf_costs(self, cost_type, nonleaf_state_weights, control_weights):
        if cost_type == "Quadratic":
            for i in range(self.__tree.num_nonleaf_nodes):
                self.__list_of_nonleaf_costs[i] = core_costs.QuadraticNonleaf(nonleaf_state_weights, control_weights)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    def with_all_leaf_costs(self, cost_type, leaf_state_weights):
        if cost_type == "Quadratic":
            for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
                self.__list_of_leaf_costs[i] = core_costs.QuadraticLeaf(leaf_state_weights)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    # Limits -----------------------------------------------------------------------------------------------------------

    def with_markovian_constraints(self, ordered_list_of_constraint_types, ordered_list_of_state_constraint_weights,
                              ordered_list_of_control_constraint_weights):
        # check same number of items in lists
        if sum((len(ordered_list_of_constraint_types),
                len(ordered_list_of_state_constraint_weights),
                len(ordered_list_of_control_constraint_weights))) / 3 != len(ordered_list_of_state_constraint_weights):
            raise ValueError("Markovian constraints lists provided are not of equal sizes")
        # check that scenario tree is Markovian
        # if self.__tree.is_markovian:
        #     for i in range(1, self.__tree.num_nodes):
        #         if ordered_list_of_constraint_types[self.__tree.value_at_node(i)] == "Rectangle":
        #             self.__list_of_nonleaf_costs[i] = core_costs.QuadraticNonleaf(
        #                 ordered_list_of_state_weights[self.__tree.value_at_node(i)],
        #                 ordered_list_of_control_weights[self.__tree.value_at_node(i)])
        #         else:
        #             raise ValueError("cost type '%s' not supported" % ordered_list_of_cost_types[i])
        #     return self
        # else:
        #     raise TypeError("costs provided as Markovian, scenario tree provided is not Markovian")

    def with_all_nonleaf_constraints(self, cost_type, nonleaf_state_weights, control_weights):
        if cost_type == "Quadratic":
            for i in range(self.__tree.num_nonleaf_nodes):
                self.__list_of_nonleaf_costs[i] = core_costs.QuadraticNonleaf(nonleaf_state_weights, control_weights)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    def with_all_leaf_constraints(self, cost_type, leaf_state_weights):
        if cost_type == "Quadratic":
            for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
                self.__list_of_leaf_costs[i] = core_costs.QuadraticLeaf(leaf_state_weights)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)):

    # Risks ------------------------------------------------------------------------------------------------------------

    def with_all_risks(self, risk_type, alpha):
        if risk_type == "AVaR":
            for i in range(self.__tree.num_nonleaf_nodes):
                self.__list_of_risks[i] = core_risks.AVaR(alpha, self.__tree.conditional_probabilities_of_children(i))
            return self
        else:
            raise ValueError("risk type '%s' not supported" % risk_type)

    # Class ------------------------------------------------------------------------------------------------------------

    def __str__(self):
        return f"RAOCP\n+ Nodes: {self.__tree.num_nodes}\n" \
               f"+ {self.__list_of_nonleaf_costs[0]}\n" \
               f"+ {self.__list_of_risks[0]}"

    def __repr__(self):
        return f"RAOCP with {self.__tree.num_nodes} nodes, " \
               f"with root cost: {type(self.__list_of_nonleaf_costs[0]).__name__}, " \
               f"with root risk: {type(self.__list_of_risks[0]).__name__}."
