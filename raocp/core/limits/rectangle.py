import numpy as np


class RectangleNonleaf:
    """
    A rectangle limit for any nonleaf node
    """

    def __init__(self, nonleaf_state_limit_weights, control_limit_weights, positive_limit, negative_limit):
        """
        :param nonleaf_state_limit_weights: nonleaf node state limit matrix (gamma_x^i)
        :param control_limit_weights: input limit matrix or scalar (gamma_u^i)
        """
        self.__nonleaf_state_limit_weights = nonleaf_state_limit_weights
        self.__control_limit_weights = control_limit_weights
        self.__positive_limit = positive_limit
        self.__negative_limit = negative_limit

    def get_cost_value(self, state, control):
        """For calculating nonleaf cost"""
        if state.shape[0] != self.__nonleaf_state_weights.shape[0]:
            raise ValueError("quadratic cost input nonleaf state dimension does not match state weight matrix")
        if isinstance(self.__control_weights, np.ndarray):
            if control.shape[0] != self.__control_weights.shape[0]:
                raise ValueError("quadratic cost input control dimension does not match control weight matrix")
            self.__most_recent_cost_value = state.T @ self.__nonleaf_state_weights @ state \
                + control.T @ self.__control_weights @ control
        elif isinstance(self.__control_weights, int):
            self.__most_recent_cost_value = state.T @ self.__nonleaf_state_weights @ state \
                + control.T @ control * self.__control_weights
        else:
            raise ValueError("control weights type '%s' not supported" % type(self.__control_weights).__name__)
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
        if leaf_state_weights.shape[0] != leaf_state_weights.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__leaf_state_weights = leaf_state_weights
        self.__most_recent_cost_value = None

    def get_cost_value(self, state):
        """For calculating leaf cost"""
        if state.shape[0] != self.__leaf_state_weights.shape[0]:
            raise ValueError("quadratic cost input leaf state dimension does not match state weight matrix")
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
