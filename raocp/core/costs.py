import numpy as np


class QuadraticNonleaf:
    """
    A quadratic cost item for any nonleaf node
    """

    def __init__(self, state_weights, control_weights):
        """
        :param state_weights: nonleaf node state cost matrix (Q)
        :param control_weights: input cost matrix or scalar (R)
        """
        if state_weights.shape[0] != state_weights.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__state_weights = state_weights
        self.__control_weights = control_weights
        self.__most_recent_cost_value = None

    def get_cost_value(self, state, control):
        """For calculating nonleaf cost"""
        if state.shape[0] != self.__state_weights.shape[0]:
            raise ValueError("quadratic cost input nonleaf state dimension does not match state weight matrix")
        if isinstance(self.__control_weights, np.ndarray):
            if control.shape[0] != self.__control_weights.shape[0]:
                raise ValueError("quadratic cost input control dimension does not match control weight matrix")
            self.__most_recent_cost_value = state.T @ self.__state_weights @ state \
                + control.T @ self.__control_weights @ control
        elif isinstance(self.__control_weights, int):
            self.__most_recent_cost_value = state.T @ self.__state_weights @ state \
                 + control.T @ control * self.__control_weights
        else:
            raise ValueError("control weights type '%s' not supported" % type(self.__control_weights).__name__)
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def state_weights(self):
        return self.__state_weights

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

    def __init__(self, state_weights):
        """
        :param state_weights: leaf node state cost matrix (Pf)
        """
        if state_weights.shape[0] != state_weights.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__state_weights = state_weights
        self.__most_recent_cost_value = None

    def get_cost_value(self, state):
        """For calculating leaf cost"""
        if state.shape[0] != self.__state_weights.shape[0]:
            raise ValueError("quadratic cost input leaf state dimension does not match state weight matrix")
        self.__most_recent_cost_value = state.T @ self.__state_weights @ state
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def state_weights(self):
        return self.__state_weights

    @property
    def most_recent_cost_value(self):
        return self.__most_recent_cost_value[0, 0]

    def __str__(self):
        return f"Cost item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Cost item; type: {type(self).__name__}"
