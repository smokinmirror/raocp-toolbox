import numpy as np
import raocp.core.cones as core_cones


class AVaR:
    """
    Risk item: Average Value at Risk class
    """

    def __init__(self, alpha, probabilities):
        """
        :param alpha: AVaR risk parameter
        :param probabilities: list of probabilities of future events

        Note: ambiguity sets of coherent risk measures can be expressed by conic inequalities,
                defined by a tuple (E, F, cone, b)
        """
        if 0 <= alpha <= 1:
            self.__alpha = alpha
        else:
            raise ValueError("alpha value '%d' not supported" % alpha)
        self.__num_children = len(probabilities)
        self.__children_probabilities = np.asarray(probabilities).reshape(self.__num_children, 1)

        self.__matrix_e = None  # coefficient matrix of mu
        self.__matrix_f = None  # coefficient matrix of nu
        self.__cone = None
        self.__vector_b = None
        self.__make_e_f_cone_b()

    def __make_e_f_cone_b(self):
        eye = np.eye(self.__num_children)
        self.__matrix_e = np.vstack((self.__alpha*eye, -eye, np.ones((1, self.__num_children))))
        self.__matrix_f = np.zeros((0, self.__num_children))
        self.__cone = core_cones.Cartesian([core_cones.NonnegativeOrthant(dimension=2 * self.__num_children),
                                            core_cones.Zero(dimension=1)])
        self.__vector_b = np.vstack((self.__children_probabilities, np.zeros((self.__num_children, 1)), 1))

    # GETTERS
    @property
    def alpha(self):
        """AVaR risk parameter alpha"""
        return self.__alpha

    @property
    def matrix_e(self):
        """Ambiguity set matrix E"""
        return self.__matrix_e

    @property
    def matrix_f(self):
        """Ambiguity set matrix F"""
        return self.__matrix_f

    @property
    def cone(self):
        """Ambiguity set cone"""
        return self.__cone

    @property
    def vector_b(self):
        """Ambiguity set vector b"""
        return self.__vector_b

    def __str__(self):
        return f"Risk item; type: {type(self).__name__}, alpha: {self.__alpha}; cone: {self.__cone.types}"

    def __repr__(self):
        return f"Risk item; type: {type(self).__name__}, alpha: {self.__alpha}; cone: {self.__cone.types}"
