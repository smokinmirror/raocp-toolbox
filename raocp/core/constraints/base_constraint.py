import numpy as np


class Constraint:
    """
    Base class for constraints
    """
    def __init__(self, node_type):
        self.__node_type = node_type
        self.__state_size = None
        self.__control_size = None
        self.__state_matrix = None
        self.__control_matrix = None
        self.__state_matrix_transposed = None
        self.__control_matrix_transposed = None

    def project(self, vector):
        pass

    # GETTERS
    @property
    def is_active(self):
        raise Exception("Base constraint accessed - actual constraint must not be setup")

    @property
    def node_type(self):
        return self.__node_type

    @property
    def state_size(self):
        return self.__state_size

    @property
    def control_size(self):
        return self.__control_size

    @property
    def state_matrix(self):
        return self.__state_matrix

    @property
    def control_matrix(self):
        return self.__control_matrix

    @property
    def state_matrix_transposed(self):
        if self.__state_matrix_transposed is None:
            raise Exception("Constraint state matrix transpose called but is None")
        else:
            return self.__state_matrix_transposed

    @property
    def control_matrix_transposed(self):
        if self.__control_matrix_transposed is None:
            raise Exception("Constraint control matrix transpose called but is None")
        else:
            return self.__control_matrix_transposed

    # SETTERS
    @state_size.setter
    def state_size(self, size):
        self.__state_size = size
        if self.__node_type.is_nonleaf and self.__control_size is not None:
            self._set_matrices()
            self._get_transpose()
        elif self.__node_type.is_nonleaf and self.__control_size is None:
            pass
        elif self.__node_type.is_leaf:
            self.__control_size = 0
            self._set_matrices()
            self._get_transpose()
        else:
            raise Exception("Node type missing")

    @control_size.setter
    def control_size(self, size):
        self.__control_size = size
        if self.__node_type.is_nonleaf and self.__state_size is not None:
            self._set_matrices()
            self._get_transpose()
        elif self.__node_type.is_nonleaf and self.__state_size is None:
            pass
        elif self.__node_type.is_leaf:
            raise Exception("Attempt to set control size on leaf node")
        else:
            raise Exception("Node type missing")

    def _set_matrices(self):
        pass

    def _get_transpose(self):
        if self.__node_type.is_nonleaf:
            self.__state_matrix_transposed = np.transpose(self.state_matrix)
            self.__control_matrix_transposed = np.transpose(self.control_matrix)
        elif self.__node_type.is_leaf:
            self.__state_matrix_transposed = np.transpose(self.state_matrix)
        else:
            raise Exception("Node type missing")

    @state_matrix.setter
    def state_matrix(self, matrix):
        self.__state_matrix = matrix

    @control_matrix.setter
    def control_matrix(self, matrix):
        if self.__node_type.is_nonleaf:
            self.__control_matrix = matrix
        elif self.__node_type.is_leaf:
            raise Exception("Attempt to set control constraint matrix of leaf node")
        else:
            raise Exception("Node type missing")

    # extras
    def __str__(self):
        return f"Base constraint"

    def __repr__(self):
        return f"Base constraint"
