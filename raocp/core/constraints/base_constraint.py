

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

    # SETTERS
    @state_size.setter
    def state_size(self, size):
        self.__state_size = size
        if self.__node_type.is_nonleaf and self.__control_size is not None:
            self._set_matrices()
        elif self.__node_type.is_nonleaf and self.__control_size is None:
            pass
        elif self.__node_type.is_leaf:
            self.__control_size = 0
            self._set_matrices()
        else:
            raise Exception("Node type missing")

    @control_size.setter
    def control_size(self, size):
        self.__control_size = size
        if self.__node_type.is_nonleaf and self.__state_size is not None:
            self._set_matrices()
        elif self.__node_type.is_nonleaf and self.__state_size is None:
            pass
        elif self.__node_type.is_leaf:
            raise Exception("Attempt to set control size on leaf node")
        else:
            raise Exception("Node type missing")

    def _set_matrices(self):
        pass

    @state_matrix.setter
    def state_matrix(self, matrix):
        self.__state_matrix = matrix

    @control_matrix.setter
    def control_matrix(self, matrix):
        self.__control_matrix = matrix

    # extras
    def __str__(self):
        return f"Base constraint"

    def __repr__(self):
        return f"Base constraint"
