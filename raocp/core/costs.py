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

    def get_cost(self, x, u=None):
        """For calculating cost at any node
        If an input (u) is not given, then it is a leaf node cost = x' Pf x.
        If an input (u) is given, then it is a nonleaf node cost = x' Q x + u' R u.
        """
        if u is None:
            cost = x.T @ self.__Pf @ x
            return cost[0, 0]
        else:
            cost = x.T @ self.__Q @ x + u.T @ self.__R @ u
            return cost[0, 0]

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
