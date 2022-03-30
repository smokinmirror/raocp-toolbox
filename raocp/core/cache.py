import numpy as np
import scipy as sp
import raocp.core.problem_spec as ps


class Cache:
    """
    Oracle of functions for solving RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: ps.RAOCP, initial_state):
        self.__raocp = problem_spec
        self.__initial_state = initial_state
        self.__x = [np.empty(0) * self.__raocp.tree.num_nodes()]
        self.__u = [np.empty(0) * self.__raocp.tree.num_nonleaf_nodes()]
        self.__y = [np.empty(0) * self.__raocp.tree.num_nodes()-1]
        self.__s = [None, np.empty(0) * self.__raocp.tree.num_nodes()-1]
        self.__t = [None, np.empty(0) * self.__raocp.tree.num_nodes()-1]
        self.__num_prim_parts = 5
        self.__w1 = []
        self.__w2 = []
        self.__w3 = []
        self.__w4 = []
        self.__w5 = []
        self.__w6 = []
        self.__w7 = []
        self.__w8 = []
        self.__w9 = []
        self.__num_dual_parts = 9

    # prox_f

    def project_on_S1(self):
        """
        z1 = [[x0 ... xN], [u0 ... u(N-1)]]
        :returns: projection of x, u onto space S_1
        """
        pass

    def project_on_S2(self):
        """
        z2 = [[y0 ... x(N-1)], [s1 ... sN], [t1 ... tN]]
        :returns: nothing
        """
        for i in range(self.__raocp.num_nonleaf_nodes):
            children = self.__raocp.tree.children_of(i)
            eye = np.eye(len(children))
            zeros = np.zeros(eye.shape)
            kernel = np.array([[self.__raocp.risk_item_at_node(i).E.T, -eye, -eye],
                               [self.__raocp.risk_item_at_node(i).F.T, zeros, zeros]])
            null = sp.linalg.null_space(kernel)
            vector = np.vstack((self.__z_prim[i, self.__y],
                                self.__z_prim[children, self.__s],
                                self.__z_prim[children, self.__t]))
            proj_z2 = null @ np.linalg.lstsq(null, vector, rcond=None)[0]
            self.__z_prim_update[i, self.__y] = np.array(proj_z2[0:2 * children + 1])
            self.__z_prim_update[children, self.__s] = np.array(proj_z2[2 * children + 1:3 * children + 1])
            self.__z_prim_update[children, self.__t] = np.array(proj_z2[3 * children + 1:4 * children + 1])

    def proximal_f(self, z):
        self.__z_prim = z
        self.project_on_S1()
        self.project_on_S2()
        # s0 ?
        return self.__z_prim_update

    # L / L*

    def L(self, z):
        pass

    def L_adjoint(self):
        pass

    # prox_g*

    def moreau_decomposition(self):
        pass

    def precomposition(self):
        pass

    def add_c(self):
        pass

    def sub_c(self):
        pass

    def projection_on_cones(self):
        pass

    def prox_g_ast(self):
        pass
