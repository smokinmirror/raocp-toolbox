import numpy as np


class Constraint:
    """
    Base class for constraints
    """
    def __init__(self):
        self.__state_size = None
        self.__control_size = None
        self.__state_matrix = None
        self.__control_matrix = None
        self.__constrained_vector = None

    @property
    def is_active(self):
        raise Exception("Base constraint accessed - actual constraint must not be setup")

    def project(self, vector):
        pass

    # GETTERS
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
    def set_state(self, state_size):
        self.__state_size = state_size
        self.__constrained_state = np.zeros((self.__state_size, 1))

    def set_control(self, control_size):
        self.__control_size = control_size
        self.__constrained_control = np.zeros((self.__control_size, 1))

    def set_state_matrix(self, state_matrix):
        self.__state_matrix = state_matrix

    def set_control_matrix(self, control_matrix):
        self.__control_matrix = control_matrix

    # extras
    def __str__(self):
        return f"Base constraint"

    def __repr__(self):
        return f"Base constraint"