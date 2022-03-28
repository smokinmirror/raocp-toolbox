import numpy as np


class Uni:
    """
    The universe cone (R^n)
    """

    def __init__(self):
        self.__dimension = 0

    def project_onto_cone(self, x):
        self.__dimension = x.size
        return x

    def project_onto_dual(self, x):  # the dual of universe is zero
        self.__dimension = x.size
        shape = x.shape
        proj_x = np.zeros(self.__dimension).reshape(shape)
        return proj_x

    # GETTERS
    @property
    def type(self):
        """Universe type"""
        return "Uni"

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

    def project_onto_cone(self, x):
        self.__dimension = x.size
        shape = x.shape
        proj_x = np.zeros(self.__dimension).reshape(shape)
        return proj_x

    def project_onto_dual(self, x):  # the dual of zero is universe
        self.__dimension = x.size
        return x

    # GETTERS
    @property
    def type(self):
        """Zero type"""
        return "Zero"

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

    def project_onto_cone(self, x):
        self.__dimension = x.size
        proj_x = np.empty((self.__dimension, 1))
        for i in range(self.__dimension):
            proj_x[i] = max(0, x[i])
        return proj_x

    def project_onto_dual(self, x):  # this cone is self dual
        return NonnegOrth.project_onto_cone(self, x)

    # GETTERS
    @property
    def type(self):
        """Nonnegative Orthant type"""
        return "NonnegOrth"

    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class SOC:
    """
    The second order cone (N^n_2)
    """

    def __init__(self):
        self.__dimension = 0

    def project_onto_cone(self, x):
        self.__dimension = x.size
        proj_x = x
        r = x[-1]
        s = np.delete(x, -1)  # returns row vector
        s_2norm = np.linalg.norm(s)
        if s_2norm <= r:
            pass  # proj_x = x
        elif s_2norm <= -r:
            proj_x = np.zeros(self.__dimension).reshape((self.__dimension, 1))
        else:
            proj_r = (s_2norm + r) / 2
            proj_s = proj_r * (s/s_2norm)
            proj_x = np.concatenate((proj_s, proj_r)).reshape((self.__dimension, 1))
        return proj_x

    def project_onto_dual(self, x):  # this cone is self dual
        return SOC.project_onto_cone(self, x)

    # GETTERS
    @property
    def type(self):
        """Second Order Cone type"""
        return "SOC"

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
        self.__dimensions = []
        self.__cones = cones
        self.__num_cones = len(cones)

    def project_onto_cone(self, x):
        size = []
        proj_x = []
        for i in range(len(x)):
            size.append(x[i].size)
            proj_x.append(self.__cones[i].project_onto_cone(x[i]))

        self.__dimension = sum(size)
        self.__dimensions = size
        return proj_x

    def project_onto_dual(self, x):
        size = []
        proj_x = []
        for i in range(len(x)):
            size.append(x[i].size)
            proj_x.append(self.__cones[i].project_onto_dual(x[i]))

        self.__dimension = sum(size)
        self.__dimensions = size
        return proj_x

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

    @property
    def dimensions(self):
        """List of the dimensions of each cone"""
        return self.__dimensions
