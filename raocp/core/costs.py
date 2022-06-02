import numpy as np


class Quadratic:
    """
    A quadratic cost item for any node
    """

    def __init__(self, node_type, state_weights, control_weights=None):
        """
        :param node_type: node class for indicating type
        :param state_weights: nonleaf node state cost matrix (Q)
        :param control_weights: input cost matrix or scalar (R)
        """
        self.__node_type = node_type
        self._check_control_weights(control_weights)
        if state_weights.shape[0] != state_weights.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__state_weights = state_weights
        self.__control_weights = control_weights
        self.__most_recent_cost_value = None

    def _check_control_weights(self, weights):
        if self.__node_type.is_nonleaf and weights is None:
            raise Exception("No control weights provided for a nonleaf node")
        if self.__node_type.is_leaf and weights is not None:
            raise Exception("Control weights provided for a leaf node")

    def get_cost_value(self, state, control=None):
        self._check_control_weights(control)
        # calculate node cost depending on type
        if self.__node_type.is_nonleaf:
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
        elif self.__node_type.is_leaf:
            if state.shape[0] != self.__state_weights.shape[0]:
                raise ValueError("quadratic cost input leaf state dimension does not match state weight matrix")
            self.__most_recent_cost_value = state.T @ self.__state_weights @ state
        else:
            raise Exception("Node type missing")
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def node_type(self):
        return self.__node_type

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
