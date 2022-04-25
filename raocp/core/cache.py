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
        self.__control_size = self.__raocp.control_dynamics_at_node(1).shape[1]
        # Chambolle-Pock primal
        self.__states = [np.zeros((self.__state_size, 0))] * self.__raocp.tree.num_nodes  # x
        self.__controls = [np.zeros((self.__control_size, 0))] * self.__raocp.tree.num_nonleaf_nodes  # u
        self.__dual_risk_variable_y = [None] * self.__raocp.tree.num_nonleaf_nodes  # y
        self.__epigraphical_relaxation_variable_s = [np.zeros(0)] * self.__raocp.tree.num_nodes  # s
        self.__epigraphical_relaxation_variable_t = [None] + [np.zeros(0)] * (self.__raocp.tree.num_nodes - 1)  # tau
        self.__num_primal_parts = 5
        # Chambolle-Pock dual
        self.__dual_part_1_nonleaf = [np.zeros(0)] * self.__raocp.tree.num_nonleaf_nodes
        self.__dual_part_2_nonleaf = [np.zeros(0)] * self.__raocp.tree.num_nonleaf_nodes
        self.__dual_part_3_nonleaf = [np.zeros(0)] * self.__raocp.tree.num_nonleaf_nodes
        self.__dual_part_4_nonleaf = [np.zeros(0)] * self.__raocp.tree.num_nonleaf_nodes
        self.__dual_part_5_nonleaf = [np.zeros(0)] * self.__raocp.tree.num_nonleaf_nodes
        self.__dual_part_6_nonleaf = [np.zeros(0)] * self.__raocp.tree.num_nonleaf_nodes
        self.__dual_part_7_leaf = [None] * self.__raocp.tree.num_nonleaf_nodes \
            + [np.zeros(0)] * (self.__raocp.tree.num_nodes - self.__raocp.tree.num_nonleaf_nodes)
        self.__dual_part_8_leaf = [None] * self.__raocp.tree.num_nonleaf_nodes \
            + [np.zeros(0)] * (self.__raocp.tree.num_nodes - self.__raocp.tree.num_nonleaf_nodes)
        self.__dual_part_9_leaf = [None] * self.__raocp.tree.num_nonleaf_nodes \
            + [np.zeros(0)] * (self.__raocp.tree.num_nodes - self.__raocp.tree.num_nonleaf_nodes)
        self.__num_dual_parts = 9
        # S1 projection
        self.__P = [np.zeros((0, 0))] * self.__raocp.tree.num_nodes
        self.__q = [np.zeros(0)] * self.__raocp.tree.num_nodes
        self.__K = [np.zeros((0, 0))] * self.__raocp.tree.num_nonleaf_nodes
        self.__d = [np.zeros(0)] * self.__raocp.tree.num_nonleaf_nodes
        self.__inverse_of_modified_control_dynamics = [np.zeros((0, 0))] * self.__raocp.tree.num_nonleaf_nodes
        self.__sum_of_dynamics = [np.zeros((0, 0))] * self.__raocp.tree.num_nodes
        # S2 projection
        self.__pseudoinverse_of_null = [np.zeros((0, 0))] * self.__raocp.tree.num_nonleaf_nodes
        # populate arrays
        self.__offline()

    # OFFLINE ##########################################################################################################

    def __offline(self):
        """
        Upon creation of Cache class, populate pre-computable arrays
        """
        for i in range(self.__raocp.tree.num_nonleaf_nodes):
            self.__dual_risk_variable_y[i] = np.zeros((2 * self.__raocp.tree.children_of(i).size + 1, 0))

        # S1 projection
        for i in range(self.__raocp.tree.num_nonleaf_nodes, self.__raocp.tree.num_nodes):
            self.__P[i] = np.eye(self.__state_size)

        state_eye = np.eye(self.__state_size)
        input_eye = np.eye(self.__control_size)
        for i in reversed(range(self.__raocp.tree.num_nonleaf_nodes)):
            sum_for_modified_control_dynamics = 0
            sum_for_K = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_modified_control_dynamics += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                    @ self.__raocp.control_dynamics_at_node(j)
                sum_for_K += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                    @ self.__raocp.state_dynamics_at_node(j)

            choleskey_of_modified_control_dynamics = np.linalg.cholesky(input_eye + sum_for_modified_control_dynamics)
            inverse_of_choleskey_of_modified_control_dynamics = np.linalg.inv(choleskey_of_modified_control_dynamics)
            self.__inverse_of_modified_control_dynamics[i] = inverse_of_choleskey_of_modified_control_dynamics.T \
                @ inverse_of_choleskey_of_modified_control_dynamics
            self.__K[i] = - self.__inverse_of_modified_control_dynamics[i] @ sum_for_K  # not correct
            sum_for_P = 0
            for j in self.__raocp.tree.children_of(i):
                self.__sum_of_dynamics[j] = self.__raocp.state_dynamics_at_node(j) \
                                    + self.__raocp.control_dynamics_at_node(j) @ self.__K[i]
                sum_for_P += self.__sum_of_dynamics[j].T @ self.__P[j] @ self.__sum_of_dynamics[j]

            self.__P[i] = state_eye + self.__K[i].T @ self.__K[i] + sum_for_P

        # S2 projection
        for i in range(self.__raocp.tree.num_nonleaf_nodes):
            eye = np.eye(len(self.__raocp.tree.children_of(i)))
            zeros = np.zeros((self.__raocp.risk_at_node(i).matrix_f.shape[1], eye.shape[0]))
            row1 = np.hstack((self.__raocp.risk_at_node(i).matrix_e.T, -eye, -eye))
            row2 = np.hstack((self.__raocp.risk_at_node(i).matrix_f.T, zeros, zeros))
            kernel = np.vstack((row1, row2))
            self.__pseudoinverse_of_null[i] = np.linalg.pinv(scipy.linalg.null_space(kernel))

    # ONLINE ###########################################################################################################

    # proximal of f ----------------------------------------------------------------------------------------------------

    def project_on_s1(self):
        """
        use dynamic programming to project (x, u) onto the set S_1
        :returns: nothing
        """
        for i in range(self.__raocp.tree.num_nonleaf_nodes, self.__raocp.tree.num_nodes):
            self.__q[i] = - 2*self.__q[i]

        for i in reversed(range(self.__raocp.tree.num_nonleaf_nodes)):
            sum_for_d = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_d += self.__raocp.control_dynamics_at_node(j).T @ self.__q[i]

            self.__d[i] = self.__inverse_of_modified_control_dynamics[i] @ \
                (self.__controls[i] - sum_for_d)
            sum_for_q = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_q += self.__sum_of_dynamics[j].T @ \
                             (self.__P[j] @ self.__raocp.control_dynamics_at_node(j) @ self.__d[i] + self.__q[j])

            self.__q[i] = - self.__states[i] + self.__K[i].T @ (self.__d[i] - self.__controls[i]) + sum_for_q

        # self.__states[0] = self.__states[0]
        for i in range(self.__raocp.tree.num_nonleaf_nodes):
            self.__controls[i] = self.__K[i] @ self.__states[i] + self.__d[i]
            for j in self.__raocp.tree.children_of(i):
                self.__states[j] = self.__sum_of_dynamics[j] @ self.__states[i] \
                                   + self.__raocp.control_dynamics_at_node(j) @ self.__d[i]

    def project_on_s2(self):
        """
        use kernels to project (y, s, t) onto the set S_2
        :returns: nothing
        """
        for i in range(self.__raocp.tree.num_nonleaf_nodes):
            children_at_i = self.__raocp.tree.children_of(i)
            s_stack = self.__epigraphical_relaxation_variable_s[children_at_i[0]]
            t_stack = self.__epigraphical_relaxation_variable_t[children_at_i[0]]
            if children_at_i.size > 1:
                for j in np.delete(children_at_i, 0):
                    s_stack = np.vstack((s_stack, self.__epigraphical_relaxation_variable_s[j]))
                    t_stack = np.vstack((t_stack, self.__epigraphical_relaxation_variable_t[j]))

            full_stack = np.vstack((self.__dual_risk_variable_y[i], s_stack, t_stack))
            projection = self.__pseudoinverse_of_null[i] @ full_stack
            self.__dual_risk_variable_y[i] = projection[0:self.__dual_risk_variable_y[i].size]
            for k in range(children_at_i.size):
                self.__epigraphical_relaxation_variable_s[children_at_i[k]] = \
                    projection[self.__dual_risk_variable_y[i].size + k]
                self.__epigraphical_relaxation_variable_t[children_at_i[k]] = \
                    projection[self.__dual_risk_variable_y[i].size + children_at_i.size + k]

    def proximal_of_f(self):
        # s0 ?
        self.project_on_s1()
        self.project_on_s2()
        return "proximal of f complete"

    # operator L and its adjoint ---------------------------------------------------------------------------------------

    def operator_ell(self):
        for i in range(self.__raocp.tree.num_nonleaf_nodes):
            self.__dual_part_1_nonleaf[i] = self.__dual_risk_variable_y[i]
            self.__dual_part_2_nonleaf[i] = self.__epigraphical_relaxation_variable_s \
                - self.__raocp.risk_at_node(i).vector_b.T @ self.__dual_risk_variable_y[i]
            self.__dual_part_3_nonleaf[i] = np.linalg.sqrtm(
                self.__raocp.nonleaf_cost_at_node(i).nonleaf_state_weights) @ self.__states[i]
            self.__dual_part_4_nonleaf[i] = np.linalg.sqrtm(self.__raocp.nonleaf_cost_at_node(i).control_weights) \
                @ self.__controls[i]
            children_at_i = self.__raocp.tree.children_of(i)
            t_stack = self.__epigraphical_relaxation_variable_t[children_at_i[0]]
            if children_at_i.size > 1:
                for j in np.delete(children_at_i, 0):
                    t_stack = np.vstack((t_stack, self.__epigraphical_relaxation_variable_t[j]))
            self.__dual_part_5_nonleaf[i] = 0.5 * t_stack
            self.__dual_part_6_nonleaf[i] = 0.5 * t_stack

        for i in range(self.__raocp.tree.num_nonleaf_nodes, self.__raocp.tree.num_nodes):
            self.__dual_part_7_leaf[i] = np.linalg.sqrtm(self.__raocp.leaf_cost_at_node(i).leaf_state_weights) \
                                         @ self.__states[i]
            self.__dual_part_8_leaf[i] = 0.5 * self.__epigraphical_relaxation_variable_s[i]
            self.__dual_part_9_leaf[i] = 0.5 * self.__epigraphical_relaxation_variable_s[i]

    def operator_ell_adjoint(self):
        for i in range(self.__raocp.tree.num_nonleaf_nodes):
            self.__dual_risk_variable_y[i] = self.__dual_part_1_nonleaf[i] - \
                                             self.__raocp.risk_at_node(i).vector_b @ self.__dual_part_2_nonleaf[i]
            self.__epigraphical_relaxation_variable_s[i] = self.__dual_part_2_nonleaf[i]
            self.__states[i] = np.linalg.sqrtm(self.__raocp.nonleaf_cost_at_node(i).nonleaf_state_weights).T \
                               @ self.__dual_part_3_nonleaf[i]
            self.__controls[i] = np.linalg.sqrtm(self.__raocp.nonleaf_cost_at_node(i).control_weights).T \
                                 @ self.__dual_part_4_nonleaf[i]
            children_at_i = self.__raocp.tree.children_of(i)
            t_stack = 0.5 * (self.__dual_part_5_nonleaf[i] + self.__dual_part_6_nonleaf[i])
            for j in range(children_at_i.size):
                self.__epigraphical_relaxation_variable_t[children_at_i[j]] = t_stack[j]

        for i in range(self.__raocp.tree.num_nonleaf_nodes, self.__raocp.tree.num_nodes):
            self.__states[i] = np.linalg.sqrtm(self.__raocp.leaf_cost_at_node(i).leaf_state_weights).T \
                               @ self.__dual_part_7_leaf[i]
            self.__epigraphical_relaxation_variable_s[i] = 0.5 * (self.__dual_part_8_leaf[i]
                                                                  + self.__dual_part_9_leaf[i])

    # proximal of g conjugate ------------------------------------------------------------------------------------------

    def moreau_decomposition(self):
        pass

    def precomposition(self):
        pass

    def add_halves(self):
        self.__dual_part_5_nonleaf -= 0.5
        self.__dual_part_6_nonleaf += 0.5
        self.__dual_part_8_leaf -= 0.5
        self.__dual_part_9_leaf += 0.5

    def subtract_halves(self):
        self.__dual_part_5_nonleaf += 0.5
        self.__dual_part_6_nonleaf -= 0.5
        self.__dual_part_8_leaf += 0.5
        self.__dual_part_9_leaf -= 0.5

    def projection_on_cones(self):
        pass

    def proximal_of_g_conjugate(self):
        pass

    # CHAMBOLLE POCK ###################################################################################################

    def chock(self, initial_state):
        """
        run the Chambolle-Pock algorithm
        """
        self.__initial_state = initial_state
