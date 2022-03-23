import numpy as np


class Cone:
    """
    Base class of cones: Universe, PosOvth, SOC, Zero
    """

    def __init__(self):
        self.project()

    def project(self):
        pass

    def project_dual(self):
        pass


class Universe(Cone):
    """
    The universe cone
    """

    def __init__(self, cone_dim):
        """
        :param cone_dim: dimensions of this cone
        """
        super().__init__()
        self.__dimensions = cone_dim
        self.check()

    def check(self):
        if self.__dimensions >= 1:
            pass
        else:
            raise ValueError("Cone dimensions should not be less than 1!")

    # GETTERS
    @property
    def type(self):
        """Universe type"""
        return f"R^{self.__dimensions}"

    @property
    def dimensions(self):
        """Universe dimensions"""
        return self.__dimensions


class PosOvth(Cone):
    """
    The positive cone
    """

    def __init__(self, cone_dim):
        """
        :param cone_dim: dimensions of this cone
        """
        super().__init__()
        self.__dimensions = cone_dim
        self.check()

    def check(self):
        if self.__dimensions >= 1:
            pass
        else:
            raise ValueError("Cone dimensions should not be less than 1!")

    # GETTERS
    @property
    def type(self):
        """PosOvth type"""
        return f"R^{self.__dimensions}_+"

    @property
    def dimensions(self):
        """PosOvth dimensions"""
        return self.__dimensions


class SOC(Cone):
    """
    The second order cone
    """

    def __init__(self, cone_dim):
        """
        :param cone_dim: dimensions of this cone
        """
        super().__init__()
        self.__dimensions = cone_dim
        self.check()

    def check(self):
        if self.__dimensions >= 1:
            pass
        else:
            raise ValueError("Cone dimensions should not be less than 1!")

    # GETTERS
    @property
    def type(self):
        """SOC type"""
        return f"SOC_{self.__dimensions}"

    @property
    def dimensions(self):
        """SOC dimensions"""
        return self.__dimensions


class Zero(Cone):
    """
    The zero cone
    """

    def __init__(self, cone_dim):
        """
        :param cone_dim: dimensions of this cone
        """
        super().__init__()
        self.__dimensions = cone_dim
        self.check()

    def check(self):
        if self.__dimensions >= 1:
            pass
        else:
            raise ValueError("Cone dimensions should not be less than 1!")

    # GETTERS
    @property
    def type(self):
        """Zero type"""
        return "{0}"

    @property
    def dimensions(self):
        """The zero cone's dimensions is always 0"""
        return 0


class Cartesian(Cone):
    """
    The Cartesian cone
    """

    def __init__(self, cone_dim, cone_1: Universe, cone_2: Zero):
        """
        :param cone_dim: dimensions of this cone
        :param cone_1: a cone class
        :param cone_2: a cone class
        """
        super().__init__()
        self.__cone_1 = cone_1
        self.__cone_2 = cone_2
        self.__dimensions = cone_dim
        self.check()

    def check(self):
        if self.__dimensions >= 1:
            pass
        else:
            raise ValueError("Cone dimensions should not be less than 1!")

    # GETTERS
    @property
    def type(self):
        """Cartesian type"""
        return f"{self.__cone_1.type}x{self.__cone_2.type}"

    @property
    def dimensions(self):
        """Cartesian dimensions"""
        return self.__dimensions
