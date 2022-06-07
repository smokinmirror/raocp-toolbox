import numpy as np
import raocp.core.constraints.base_constraint as bc


class Rectangle(bc.Constraint):
    """
    A rectangle constraint
    """
    def __init__(self, node_type, _min, _max):
        """
        :param node_type: nonleaf or leaf
        :param _min: vector of minimum values
        :param _max: vector of maximum values
        """
        super().__init__(node_type)
        self._check_constraints(_min, _max)
        self.__min = _min
        self.__max = _max

    @property
    def is_active(self):
        return True

    def _set_matrices(self):
        self.state_matrix = np.vstack((np.eye(self.state_size), np.zeros((self.control_size, self.state_size))))
        if self._Constraint__node_type.is_nonleaf:
            self.control_matrix = np.vstack((np.zeros((self.state_size, self.control_size)), np.eye(self.control_size)))

    def project(self, vector):
        self._check_input(vector)
        constrained_vector = np.zeros(vector.shape)
        for i in range(vector.size):
            constrained_vector[i] = self._constrain(vector[i], self.__min[i], self.__max[i])

        return constrained_vector

    @staticmethod
    def _check_constraints(_min, _max):
        if _min.size != _max.size:
            raise Exception("Rectangle constraint - min and max vectors sizes are not equal")
        for i in range(_min.size):
            if _min[i] is None and _max[i] is None:
                raise Exception("Rectangle constraint - both min and max constraints cannot be None")
            if _min[i] is None or _max[i] is None:
                pass
            else:
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
            raise ValueError(f"Rectangle constraint - '{value}' value cannot be constrained")

    def _check_input(self, vector):
        if vector.size != self.state_matrix.shape[0]:
            raise Exception("Rectangle constraint - input vector does not equal expected size")

    def __str__(self):
        return f"Constraint; type: {type(self).__name__}"

    def __repr__(self):
        return f"Constraint; type: {type(self).__name__}"
