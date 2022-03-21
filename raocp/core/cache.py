import numpy as np


class cache:
    """
    Oracle of functions for solving RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec):
        self.__raocp = problem_spec

    # prox_f

    def projection_on_S1(self):
        pass

    def projection_on_S2(self):
        pass

    def proximal_f(self):
        pass

    # L / L*

    def L(self):
        pass

    def L_adjoint(self):
        pass

    # prox_g*

    def moreau_decomposition(self):
        pass

    def precomposition(self):
        pass

    def add_c(self):
        pass

    def sub_c(self):
        pass

    def projection_on_cones(self):
        pass

    def prox_g_ast(self):
        pass
