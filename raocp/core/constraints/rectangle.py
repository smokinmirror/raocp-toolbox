import raocp.core.constraints.base_constraint as bc
import numpy as np


class Rectangle(bc.Constraint):
    """
    A rectangle constraint
    """
    def __init__(self, state_min=None, state_max=None, control_min=None, control_max=None):
        """
        :param state_min: vector of minimum values for the system states
        :param state_max: vector of maximum values for the system states
        :param control_min: vector of minimum values for the control actions
        :param control_max: vector of maximum values for the control actions
        """
        super().__init__()
        self._check_constraints(state_min, state_max, control_min, control_max)
        self.__state_min = state_min
        self.__state_max = state_max
        self.__control_min = control_min
        self.__control_max = control_max
        self.__state_matrix = np.vstack((np.eye(self.__state_size),
                                         np.zeros((self.__control_size, self.__control_size))))
        self.__control_matrix = np.vstack((np.zeros(self.__state_size, self.__state_size),
                                           np.eye(self.__control_size)))

    @property
    def is_active(self):
        return True

    def project(self, state, control=None):
        self._check_inputs(state, control)
        for i in state.size:
            self.__constrained_state[i] = self._constrain(state[i],
                                                          self.__state_min[i],
                                                          self.__state_max[i])

        if control is None:
            return self.__constrained_state
        else:
            for i in control.size:
                self.__constrained_control[i] = self._constrain(control[i],
                                                                self.__control_min[i],
                                                                self.__control_max[i])

            return self.__constrained_state, self.__constrained_control

    @staticmethod
    def _check_constraints(s_min, s_max, c_min, c_max):
        # s = state
        if s_min.size != s_max.size:
            raise Exception("Rectangle constraint - state min and max vectors sizes are not equal")
        for i in s_min.size:
            if s_min[i] is None and s_max[i] is None:
                raise Exception("Rectangle constraint - both min and max state constraints cannot be None")
            if s_min[i] > s_max[i]:
                raise Exception("Rectangle constraint - state min greater than max")

        # c = control
        if c_min.size != c_max.size:
            raise Exception("Rectangle constraint - control min and max vectors sizes are not equal")
        for i in c_min.size:
            if c_min[i] is None and c_max[i] is None:
                raise Exception("Rectangle constraint - both min and max control constraints cannot be None")
            if c_min[i] > c_max[i]:
                raise Exception("Rectangle constraint - control min greater than max")

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

    def _check_inputs(self, state, control):
        if state.size != self.__state_size:
            raise Exception("Rectangle constraint - input state does not equal expected state size")
        if control is not None:
            if control.size != self.__control_size:
                raise Exception("Rectangle constraint - input control does not equal expected control size")

    def __str__(self):
        return f"Constraint; type: {type(self).__name__}"

    def __repr__(self):
        return f"Constraint; type: {type(self).__name__}"
