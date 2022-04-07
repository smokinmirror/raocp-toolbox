import numpy as np
import scipy.optimize
import raocp.core.problem_spec as ps


class Cache:
    """
    Oracle of functions for solving RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: ps.RAOCP):
        self.__raocp = problem_spec
        self.__initial_state = None
        self.__state_size = self.__raocp.state_dynamics_at_node(1).shape[1]
        self.__input_size = self.__raocp.control_dynamics_at_node(1).shape[1]
        # primal
        self.__x = [np.empty(0) * self.__raocp.tree.num_nodes]
        self.__u = [np.empty(0) * self.__raocp.tree.num_nonleaf_nodes]
        self.__y = [np.empty(0) * self.__raocp.tree.num_nodes - 1, None]
        self.__s = [None, np.empty(0) * self.__raocp.tree.num_nodes - 1]
        self.__t = [None, np.empty(0) * self.__raocp.tree.num_nodes - 1]
        self.__num_prim_parts = 5
        # dual
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
        # S1 projection
        self.__P = [np.empty((0, 0))] * self.__raocp.tree.num_nodes
        self.__q = [np.empty(0)] * self.__raocp.tree.num_nodes
        self.__K = [np.empty((0, 0))] * self.__raocp.tree.num_nonleaf_nodes
        self.__d = [np.empty(0)] * self.__raocp.tree.num_nonleaf_nodes
        self.__Bhat_inv = [np.empty((0, 0))] * self.__raocp.tree.num_nonleaf_nodes
        self.__AplusBK = [np.empty((0, 0))] * self.__raocp.tree.num_nodes
        # S2 projection
        self.__null = [np.empty((0, 0))] * self.__raocp.tree.num_nonleaf_nodes
        # populate arrays
        self.__offline()

    # OFFLINE ##########################################################################################################

    def __offline(self):
        """
        Upon creation of Cache class, populate pre-computable arrays
        """

        # S1 projection
        for i in range(self.__raocp.tree.num_nonleaf_nodes, self.__raocp.tree.num_nodes):
            self.__P[i] = np.eye(self.__state_size)

        state_eye = np.eye(self.__state_size)
        input_eye = np.eye(self.__input_size)
        for i in reversed(range(self.__raocp.tree.num_nonleaf_nodes)):
            sum_B = 0
            sum_K = 0
            for j in self.__raocp.tree.children_of(i):
                sum_B += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                         @ self.__raocp.control_dynamics_at_node(j)
                sum_K += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                         @ self.__raocp.state_dynamics_at_node(j)

            Bhat_chol = np.linalg.cholesky(input_eye + sum_B)
            Bhat_chol_inv = np.linalg.inv(Bhat_chol)
            self.__Bhat_inv[i] = Bhat_chol_inv.T @ Bhat_chol_inv
            self.__K[i] = - self.__Bhat_inv[i] @ sum_K  # not correct
            sum_P = 0
            for j in self.__raocp.tree.children_of(i):
                self.__AplusBK[j] = self.__raocp.state_dynamics_at_node(j) \
                                    + self.__raocp.control_dynamics_at_node(j) @ self.__K[i]
                sum_P += self.__AplusBK[j].T @ self.__P[j] @ self.__AplusBK[j]

            self.__P[i] = state_eye + self.__K[i].T @ self.__K[i] + sum_P

        # S2 projection
        for i in range(self.__raocp.tree.num_nonleaf_nodes):
            eye = np.eye(len(self.__raocp.tree.children_of(i)))
            zeros = np.zeros((self.__raocp.risk_at_node(i).matrix_f.shape[1], eye.shape[0]))
            row1 = np.hstack((self.__raocp.risk_at_node(i).matrix_e.T, -eye, -eye))
            row2 = np.hstack((self.__raocp.risk_at_node(i).matrix_f.T, zeros, zeros))
            kernel = np.vstack((row1, row2))
            self.__null[i] = scipy.linalg.null_space(kernel)

    # ONLINE ###########################################################################################################

    # prox_f -----------------------------------------------------------------------------------------------------------

    def project_on_s1(self):
        """
        use dynamic programming to project (x, u) onto the set S_1
        :returns: nothing
        """
        pass

    def project_on_s2(self):
        """
        use kernels to project (y, s, t) onto the set S_2
        :returns: nothing
        """
        for i in range(self.__raocp.tree.num_nonleaf_nodes):
            children = self.__raocp.tree.children_of(i)
            # vector = np.vstack((self.__z_prim[i, self.__y],
            #                     self.__z_prim[children, self.__s],
            #                     self.__z_prim[children, self.__t]))
            # proj_z2 = self.__null[i] @ np.linalg.lstsq(null, vector, rcond=None)[0]
            # self.__z_prim_update[i, self.__y] = np.array(proj_z2[0:2 * children + 1])
            # self.__z_prim_update[children, self.__s] = np.array(proj_z2[2 * children + 1:3 * children + 1])
            # self.__z_prim_update[children, self.__t] = np.array(proj_z2[3 * children + 1:4 * children + 1])

    def proximal_f(self):
        # s0 ?
        self.project_on_s1()
        self.project_on_s2()
        return "prox_f complete"

    # L / L* -----------------------------------------------------------------------------------------------------------

    def L(self, z):
        pass

    def L_adjoint(self):
        pass

    # prox_g* ----------------------------------------------------------------------------------------------------------

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

    # CHAMBOLLE POCK ###################################################################################################

    def chock(self, initial_state):
        """
        run the chambolle-pock algorithm
        """
        self.__initial_state = initial_state
