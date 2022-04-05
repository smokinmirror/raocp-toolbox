import numpy as np


class QuadraticNonleaf:
    """
    A quadratic cost item for any nonleaf node
    """

    def __init__(self, nonleaf_state_weights, control_weights):
        """
        :param nonleaf_state_weights: nonleaf node state cost matrix (Q)
        :param control_weights: input cost matrix (R)
        """
        self.__nonleaf_state_weights = nonleaf_state_weights
        self.__control_weights = control_weights
        self.__most_recent_cost_value = None

    def get_cost(self, state, control):
        """For calculating nonleaf cost"""
        self.__most_recent_cost_value = state.T @ self.__nonleaf_state_weights @ state \
            + control.T @ self.__control_weights @ control
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def nonleaf_state_weights(self):
        return self.__nonleaf_state_weights

    @property
    def control_weights(self):
        return self.__control_weights

    @property
    def most_recent_cost_value(self):
        return self.__most_recent_cost_value[0, 0]

    def __str__(self):
        return f"Cost item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Cost item; type: {type(self).__name__}"


class QuadraticLeaf:
    """
    A quadratic cost item for any leaf node
    """

    def __init__(self, leaf_state_weights):
        """
        :param leaf_state_weights: leaf node state cost matrix (Pf)
        """
        self.__leaf_state_weights = leaf_state_weights
        self.__most_recent_cost_value = None

    def get_cost(self, state):
        """For calculating leaf cost"""
        self.__most_recent_cost_value = state.T @ self.__leaf_state_weights @ state
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def leaf_state_weights(self):
        return self.__leaf_state_weights

    @property
    def most_recent_cost_value(self):
        return self.__most_recent_cost_value[0, 0]

    def __str__(self):
        return f"Cost item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Cost item; type: {type(self).__name__}"
