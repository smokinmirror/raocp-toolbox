import raocp.core.constraints.base_constraint as bc
import numpy as np


class Rectangle(bc.Constraint):
    """
    A rectangle constraint
    """
    def __init__(self, _min=None, _max=None):
        """
        :param _min: vector of minimum values
        :param _max: vector of maximum values
        """
        super().__init__()
        self._check_constraints(_min, _max)
        self.__min = _min
        self.__max = _max
        self.__state_matrix = np.vstack((np.eye(self.__state_size),
                                         np.zeros((self.__control_size, self.__control_size))))
        self.__control_matrix = np.vstack((np.zeros(self.__state_size, self.__state_size),
                                           np.eye(self.__control_size)))

    @property
    def is_active(self):
        return True

    def project(self, vector):
        self._check_input(vector)
        for i in vector.size:
            self.__constrained_vector[i] = self._constrain(vector[i], self.__min[i], self.__max[i])

    @staticmethod
    def _check_constraints(_min, _max):
        if _min.size != _max.size:
            raise Exception("Rectangle constraint - min and max vectors sizes are not equal")
        for i in _min.size:
            if _min[i] is None and _max[i] is None:
                raise Exception("Rectangle constraint - both min and max constraints cannot be None")
            if _min[i] > _max[i]:
                raise Exception("Rectangle constraint - min greater than max")

    @staticmethod
    def _constrain(value, mini, maxi):
        if mini <= value <= maxi:
            return value
        elif value <= mini:
            return mini
        elif value >= maxi:
            return maxi
        else:
            raise Exception(f"Rectangle constraint - '{value}' value cannot be constrained")

    def _check_input(self, vector):
        if vector.size != self.__state_matrix.shape[0]:
            raise Exception("Rectangle constraint - input vector does not equal expected size")

    def __str__(self):
        return f"Constraint; type: {type(self).__name__}"

    def __repr__(self):
        return f"Constraint; type: {type(self).__name__}"
