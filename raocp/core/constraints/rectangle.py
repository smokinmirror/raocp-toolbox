import raocp.core.constraints.base_constraint as bc
import numpy as np


def _check_inputs(ip_min, ip_max):
    if ip_min is None and ip_max is None:
        raise Exception("Rectangle constraint - both constraints cannot be None")
    if ip_min > ip_max:
        raise Exception("Rectangle constraint - min greater than max")


class RectangleNonleaf(bc.Constraint):
    """
    A rectangle constraint for state-inputs
    """
    def __init__(self, _min=None, _max=None):
        """
        :param _min: minimum value for state-input constraint
        :param _max: maximum value for state-input constraint
        """
        super().__init__()
        _check_inputs(_min, _max)
        self.__min = _min
        self.__max = _max
        self.__state_matrix = np.ones(self.__state_size).reshape((self.__state_size, 1))
        self.__control_matrix = np.ones(self.__control_size).reshape((self.__control_size, 1))

    def project(self, state, control):
        pass

    def __str__(self):
        return f"Constraint; type: {type(self).__name__}"

    def __repr__(self):
        return f"Constraint; type: {type(self).__name__}"


class RectangleLeaf:
    """
    A rectangle constraint for states
    """
    def __init__(self, _min=None, _max=None):
        """
        :param _min: minimum value for state constraint
        :param _max: maximum value for state constraint
        """
        _check_inputs(_min, _max)
        self.__min = _min
        self.__max = _max
        self.__state_matrix = None

    def project(self):
        pass

    # GETTERS

    @property
    def state_matrix(self):
        return self.__state_matrix

    def __str__(self):
        return f"Constraint; type: {type(self).__name__}"

    def __repr__(self):
        return f"Constraint; type: {type(self).__name__}"
