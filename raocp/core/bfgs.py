import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import time
import raocp.core.cache as cache
import raocp.core.operators as ops
import raocp.core.raocp_spec as spec

import matplotlib.pyplot as plt
import tikzplotlib as tikz


class Bfgs:
    """
    BFGS accelerator
    """

    def __init__(self, memory):
        self.__memory = memory