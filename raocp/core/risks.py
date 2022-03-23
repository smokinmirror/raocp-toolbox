import numpy as np


class AVaR:
    """
    Risk item: Average Value at Risk class
    """

    def __init__(self, alpha):
        """
        :param alpha: AVaR risk parameter

        Note: ambiguity sets of coherent risk measures can be expressed by conic inequalities,
                defined by a tuple (E, F, cone, b)
        """
        self.__alpha = alpha

        self.__e = None
        self.__f = None
        self.__cone = None
        self.__b = None
        self.__make_e_f_cone_b()

    def __make_e_f_cone_b(self):


    # GETTERS
    @property
    def alpha(self):
        """AVaR risk parameter alpha"""
        return self.__alpha

    @property
    def e(self):
        """ambiguity set matrix E"""
        return self.__e

    @property
    def f(self):
        """ambiguity set matrix F"""
        return self.__e

    @property
    def cone(self):
        """ambiguity set cone"""
        return self.__cone

    @property
    def b(self):
        """ambiguity set vector b"""
        return self.__b
