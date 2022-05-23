import raocp.core.constraints.base_constraint as bc
import numpy as np


class No(bc.Constraint):
    """
    For no constraints
    """
    def __init__(self):
        super().__init__()

    def project(self, state, control=None):
        if control is None:
            return state
        else:
            return state, control

    def set_state(self, state_size):
        bc.Constraint.set_state(self, state_size)
        bc.Constraint.set_state_matrix(self, np.ones(self.state_size).reshape((1, self.state_size)))
        return self

    def set_control(self, control_size):
        bc.Constraint.set_control(self, control_size)
        bc.Constraint.set_control_matrix(self, np.ones(self.control_size).reshape((1, self.control_size)))
        return self
