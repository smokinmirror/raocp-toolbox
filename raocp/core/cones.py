import numpy as np


def _check_dimension(cone_type, cone_dimension, vector):
    """
    Function for checking cone dimensions against given vector

    If dimensions match, return vector size.
    If dimensions do not match, raise error.
    """
    vector_dimension = vector.size
    if cone_dimension is None:
        cone_dimension = vector_dimension
    if cone_dimension != vector_dimension:
        raise ValueError("%s cone dimension error: cone dimension = %d, input vector dimension = %d"
                         % (cone_type, cone_dimension, vector_dimension))
    else:
        return vector_dimension


class Real:
    """
    A cone of reals of dimension n (R^n)
    """

    def __init__(self, dimension=None):
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = vector.copy()
        return projection

    def project_onto_dual(self, vector):  # the dual of real is zero
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = np.zeros(self.__dimension).reshape(self.__shape)
        return projection

    # GETTERS
    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class Zero:
    """
    A zero cone ({0})
    """

    def __init__(self, dimension=None):
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = np.zeros(self.__dimension).reshape(self.__shape)
        return projection

    def project_onto_dual(self, vector):  # the dual of zero is real
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = vector.copy()
        return projection

    # GETTERS
    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class NonnegativeOrthant:
    """
    A nonnegative orthant cone of dimension n (R^n_+)
    """

    def __init__(self, dimension=None):
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = np.empty(self.__shape)
        for i in range(self.__dimension):
            projection[i] = max(0, vector[i])
        return projection

    def project_onto_dual(self, vector):  # this cone is self dual
        return NonnegativeOrthant.project(self, vector)

    # GETTERS
    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class SecondOrderCone:
    """
    A second order cone (N^n_2)
    """

    def __init__(self, dimension=None):
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        last_part = vector[-1].reshape(1, 1)
        first_part = vector[0:-1]
        two_norm_of_first_part = np.linalg.norm(first_part)
        if two_norm_of_first_part <= last_part:
            projection = vector.copy()
            return projection
        elif two_norm_of_first_part <= -last_part:
            projection = np.zeros(shape=self.__shape)
            return projection
        else:
            projection_of_last_part = (two_norm_of_first_part + last_part) / 2
            projection_of_first_part = projection_of_last_part * (first_part/two_norm_of_first_part)
            projection = np.concatenate((projection_of_first_part,
                                         projection_of_last_part)).reshape(self.__shape)
            return projection

    def project_onto_dual(self, vector):  # this cone is self dual
        return SecondOrderCone.project(self, vector)

    # GETTERS
    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class Cartesian:
    """
    The Cartesian product of cones (cone x cone x ...)
    """

    def __init__(self, cones):
        """
        :param cones: ordered list of cones
        """
        self.__cones = cones
        self.__num_cones = len(cones)
        self.__dimension = None
        self.__dimensions = [None] * self.__num_cones

    def project(self, list_of_vectors):
        projection = []
        for i in range(self.__num_cones):
            self.__dimensions[i] = _check_dimension(type(self.__cones[i]),
                                                    self.__cones[i].dimension,
                                                    list_of_vectors[i])
            projection.append(self.__cones[i].project(list_of_vectors[i]))

        self.__dimension = sum(self.__dimensions)
        return projection

    def project_onto_dual(self, list_of_vectors):
        projection = []
        for i in range(self.__num_cones):
            self.__dimensions[i] = _check_dimension(type(self.__cones[i]),
                                                    self.__cones[i].dimension,
                                                    list_of_vectors[i])
            projection.append(self.__cones[i].project_onto_dual(list_of_vectors[i]))

        self.__dimension = sum(self.__dimensions)
        return projection

    # GETTERS
    @property
    def types(self):
        """Cartesian product of cones type"""
        product = type(self.__cones[0]).__name__
        for i in self.__cones[1:]:
            product = product + " x " + type(i).__name__
        return product

    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension

    @property
    def dimensions(self):
        """List of the dimensions of each cone"""
        return self.__dimensions

    @property
    def num_cones(self):
        """Number of cones that make up Cartesian cone"""
        return self.__num_cones
