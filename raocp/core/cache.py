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
        self.__x = 0
        self.__u = 1
        self.__y = 2
        self.__s = 3
        self.__t = 4
        self.__z_prim = np.array([[None] * 5] * self.__raocp.tree.num_nodes(), dtype=object)
        self.__z_prim_update = np.array([[None] * 5] * self.__raocp.tree.num_nodes(), dtype=object)
        self.__z_dual = np.array([[None] * 5] * self.__raocp.tree.num_nodes(), dtype=object)
        self.__z_dual_update = np.array([[None] * 5] * self.__raocp.tree.num_nodes(), dtype=object)

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
        :returns: projection of y, s, t onto linear space S_2
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
        for i in range(self.__raocp.num_nonleaf_nodes):
            w1 = i  # what

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
