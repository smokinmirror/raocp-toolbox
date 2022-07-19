import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import time
import raocp.core.cache as cache
import raocp.core.operators as ops
import raocp.core.raocp_spec as spec


class SuperMann:
    """
    SuperMann accelerator
    """

    def __init__(self, operator):
        self.T = operator
        self.__c0 = 0.99
        self.__c1 = 0.99
        self.__c2 = 0.99
        self.__beta = 0.5
        self.__sigma = 0.1
        self.__lamda = 1.95
        self.__eta = [None] * 2
        self.__w = [None] * 2
        self.__pos_k = 0
        self.__pos_kplus1 = 1
        self.__r_safe = None
        self.__counter_k = 0

    def accelerate(self, x):
        self.T(x)
        if ...
