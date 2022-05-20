import numpy as np


class RectangleNonleaf:
    """
    A rectangle constraint for state-inputs on nonleaf nodes
    """
    def __init__(self, _min, _max):
        """
        :param _min: minimum value for state-input constraint
        :param _max: maximum value for state-input constraint
        """
        self.__min = _min
        self.__max = _max
        self.__state_matrix = None
        self.__control_matrix = None

    def project(self):
        pass

    # GETTERS
    @property
    def state_matrix(self):
        return self.__state_matrix

    @property
    def control_matrix(self):
        return self.__control_matrix

    def __str__(self):
        return f"Constraint; type: {type(self).__name__}"

    def __repr__(self):
        return f"Constraint; type: {type(self).__name__}"


class RectangleLeaf:
    """
    A rectangle constraint for states on leaf nodes
    """
    def __init__(self, _min, _max):
        """
        :param _min: minimum value for state constraint
        :param _max: maximum value for state constraint
        """
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
