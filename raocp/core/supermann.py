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
