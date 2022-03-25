import numpy as np


class Uni:
    """
    The universe cone (R^n)
    """

    def __init__(self):
        self.__dimension = 0

    # GETTERS
    @property
    def type(self):
        """Universe type"""
        return f"R^{self.__dimension}"

    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class NonnegOrth:
    """
    The nonnegative orthant cone (R^n_+)
    """

    def __init__(self):
        self.__dimension = 0

    def projection_onto_cone(self, x):
        self.__dimension = x.size
        proj_x = x
        for i in range(self.__dimension):
            proj_x[i] = max(0, x[i])
        return proj_x

    def projection_onto_dual(self, x):  # this cone is self dual
        self.__dimension = x.size
        proj_x = x
        for i in range(self.__dimension):
            proj_x[i] = max(0, x[i])
        return proj_x

    # GETTERS
    @property
    def type(self):
        """Nonnegative Orthant type"""
        return f"R^{self.__dimension}_+"

    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class SOC:
    """
    The second order cone ()
    """

    def __init__(self):
        self.__dimension = 0

    # GETTERS
    @property
    def type(self):
        """Second Order Cone type"""
        return f"SOC^{self.__dimension}"

    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class Zero:
    """
    The zero cone ({0})
    """

    def __init__(self):
        self.__dimension = 0

    # GETTERS
    @property
    def type(self):
        """Zero type"""
        return f"(0)^{self.__dimension}"

    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class Cart:
    """
    The Cartesian product of cones (cone x cone)
    """

    def __init__(self, cones):
        """
        :param cones: list of cones
        """
        self.__dimension = 0
        self.__cones = cones
        self.__num_cones = len(cones)

    # GETTERS
    @property
    def type(self):
        """Cartesian product of cones type"""
        product = self.__cones[0].type
        for i in self.__cones[1:]:
            product = product + " x " + i.type
        return product

    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension
