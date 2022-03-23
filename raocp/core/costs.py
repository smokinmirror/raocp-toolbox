import numpy as np


class Quadratic:
    """
    Cost item: quadratic cost for any node class
    """

    def __init__(self, Q, R, Pf, node):
        """
        :param Q: nonleaf node state cost matrix
        :param R: input cost matrix
        :param Pf: leaf node state cost matrix
        :param node: current node
        """
        self.__Q = Q
        self.__R = R
        self.__Pf = Pf
        self.__node = node

    def get_cost(self):
        pass

    # GETTERS
    @property
    def type(self):
        """Cost type"""
        return "quadratic"

    @property
    def Q(self):
        """Q"""
        return self.__Q

    @property
    def R(self):
        """R"""
        return self.__R

    @property
    def Pf(self):
        """Pf"""
        return self.__Pf

    def __str__(self):
        return f"Cost item at node {self.__node}; type: quadratic"

    def __repr__(self):
        return f"Cost item at node {self.__node}; type: quadratic"
