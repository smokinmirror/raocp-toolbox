import numpy as np
import scipy.optimize
from scipy.linalg import sqrtm
import raocp.core.problem_spec as ps
import raocp.core.cones as cones


class Cache:
    """
    Oracle of functions for solving RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: ps.RAOCP):
        self.__raocp = problem_spec
        self.__num_nodes = self.__raocp.tree.num_nodes
        self.__num_nonleaf_nodes = self.__raocp.tree.num_nonleaf_nodes
        self.__num_stages = self.__raocp.tree.num_stages
        self.__state_size = self.__raocp.state_dynamics_at_node(1).shape[1]
        self.__control_size = self.__raocp.control_dynamics_at_node(1).shape[1]

        # Chambolle-Pock
        self.__gamma = None

        # Chambolle-Pock primal
        self.__states = [np.zeros((self.__state_size, 1))] * self.__num_nodes  # x
        self.__controls = [np.zeros((self.__control_size, 1))] * self.__num_nonleaf_nodes  # u
        self.__dual_risk_variable_y = [np.zeros(0)] * self.__num_nonleaf_nodes  # y
        self.__epigraphical_relaxation_variable_s = [np.zeros(0)] * (self.__num_stages + 1)  # s
        self.__epigraphical_relaxation_variable_tau = [None] + [np.zeros(0)] * self.__num_stages  # tau
        for i in range(self.__num_nonleaf_nodes):
            self.__dual_risk_variable_y[i] = np.zeros((2 * self.__raocp.tree.children_of(i).size + 1, 1))

        for i in range(self.__num_stages + 1):
            largest_node_at_stage = max(self.__raocp.tree.nodes_at_stage(i))
            # store variables in their node number inside the stage vector for s and tau
            self.__epigraphical_relaxation_variable_s[i] = np.zeros((largest_node_at_stage + 1, 1))
            if i > 0:
                self.__epigraphical_relaxation_variable_tau[i] = np.zeros((largest_node_at_stage + 1, 1))

        # Chambolle-Pock dual / parts 3,4,5,6 stored in child nodes for convenience
        self.__dual_part = [np.zeros(1)] * (self.__num_nodes * 9)
        self.__dual_part_1_nonleaf = self.__dual_part[0: self.__num_nodes]
        self.__dual_part_2_nonleaf = self.__dual_part[self.__num_nodes * 1: self.__num_nodes * 2]
        self.__dual_part_3_nonleaf = self.__dual_part[self.__num_nodes * 2: self.__num_nodes * 3]
        self.__dual_part_4_nonleaf = self.__dual_part[self.__num_nodes * 3: self.__num_nodes * 4]
        self.__dual_part_5_nonleaf = self.__dual_part[self.__num_nodes * 4: self.__num_nodes * 5]
        self.__dual_part_6_nonleaf = self.__dual_part[self.__num_nodes * 5: self.__num_nodes * 6]
        self.__dual_part_7_leaf = self.__dual_part[self.__num_nodes * 6: self.__num_nodes * 7]
        self.__dual_part_8_leaf = self.__dual_part[self.__num_nodes * 7: self.__num_nodes * 8]
        self.__dual_part_9_leaf = self.__dual_part[self.__num_nodes * 8: self.__num_nodes * 9]
        for i in range(1, self.__num_nodes):
            self.__dual_part_3_nonleaf[i] = np.zeros((self.__state_size, 1))
            self.__dual_part_4_nonleaf[i] = np.zeros((self.__control_size, 1))
            if i >= self.__num_nonleaf_nodes:
                self.__dual_part_7_leaf[i] = np.zeros((self.__state_size, 1))

        # S1 projection
        self.__P = [np.zeros((self.__state_size, self.__state_size))] * self.__num_nodes
        self.__q = [np.zeros((self.__state_size, 1))] * self.__num_nodes
        self.__K = [np.zeros((0, 0))] * self.__num_nonleaf_nodes
        self.__d = [np.zeros(0)] * self.__num_nonleaf_nodes
        self.__inverse_of_modified_control_dynamics = [np.zeros((0, 0))] * self.__num_nonleaf_nodes
        self.__sum_of_dynamics = [np.zeros((0, 0))] * self.__num_nodes  # A+BK
        # S2 projection
        self.__s2_projection_operator = [np.zeros((0, 0))] * self.__num_nonleaf_nodes

        # cones
        self.__nonleaf_constraint_cone = [None] * self.__num_nonleaf_nodes
        self.__nonleaf_second_order_cone = [None] * self.__num_nodes
        self.__leaf_second_order_cone = [None] * self.__num_nodes
        for i in range(self.__num_nodes):
            if i < self.__num_nonleaf_nodes:
                self.__nonleaf_constraint_cone[i] = cones.Cartesian([cones.NonnegativeOrthant()] * 2)
            if i > 0:
                self.__nonleaf_second_order_cone[i] = cones.SecondOrderCone()
            if i > self.__num_nonleaf_nodes:
                self.__leaf_second_order_cone[i] = cones.SecondOrderCone()

        # populate arrays
        self.__offline()

    # OFFLINE ##########################################################################################################

    def __offline(self):
        """
        Upon creation of Cache class, populate pre-computable arrays
        """
        # S1 projection offline
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__P[i] = np.eye(self.__state_size)

        state_eye = np.eye(self.__state_size)
        control_eye = np.eye(self.__control_size)
        for i in reversed(range(self.__num_nonleaf_nodes)):
            sum_for_modified_control_dynamics = 0
            sum_for_K = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_modified_control_dynamics += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                    @ self.__raocp.control_dynamics_at_node(j)
                sum_for_K += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                    @ self.__raocp.state_dynamics_at_node(j)

            choleskey_of_modified_control_dynamics = np.linalg.cholesky(control_eye + sum_for_modified_control_dynamics)
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

        # S2 projection offline
        for i in range(self.__num_nonleaf_nodes):
            eye = np.eye(len(self.__raocp.tree.children_of(i)))
            zeros = np.zeros((self.__raocp.risk_at_node(i).matrix_f.shape[1], eye.shape[0]))
            row1 = np.hstack((self.__raocp.risk_at_node(i).matrix_e.T, -eye, -eye))
            row2 = np.hstack((self.__raocp.risk_at_node(i).matrix_f.T, zeros, zeros))
            s2_space = np.vstack((row1, row2))
            kernel = scipy.linalg.null_space(s2_space)
            pseudoinverse_of_kernel = np.linalg.pinv(kernel)
            self.__s2_projection_operator[i] = kernel @ pseudoinverse_of_kernel

    # ONLINE ###########################################################################################################

    # proximal of f ----------------------------------------------------------------------------------------------------

    def project_on_s1(self):
        """
        use dynamic programming to project (x, u) onto the set S_1
        :returns: nothing
        """
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__q[i] = - 2*self.__q[i]

        for i in reversed(range(self.__num_nonleaf_nodes)):
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
        for i in range(self.__num_nonleaf_nodes):
            self.__controls[i] = self.__K[i] @ self.__states[i] + self.__d[i]
            for j in self.__raocp.tree.children_of(i):
                self.__states[j] = self.__sum_of_dynamics[j] @ self.__states[i] \
                                   + self.__raocp.control_dynamics_at_node(j) @ self.__d[i]

    def project_on_s2(self):
        """
        use kernels to project (y, s, tau) onto the set S_2
        :returns: nothing
        """
        for i in range(self.__num_nonleaf_nodes):
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            # get children of i out of next stage of s and tau
            s_stack = self.__epigraphical_relaxation_variable_s[stage_at_children_of_i][children_of_i[0]]
            tau_stack = self.__epigraphical_relaxation_variable_tau[stage_at_children_of_i][children_of_i[0]]
            if children_of_i.size > 1:
                for j in np.delete(children_of_i, 0):
                    s_stack = np.vstack((s_stack,
                                         self.__epigraphical_relaxation_variable_s[stage_at_children_of_i][j]))
                    tau_stack = np.vstack((tau_stack,
                                           self.__epigraphical_relaxation_variable_tau[stage_at_children_of_i][j]))

            # print(i, self.__dual_risk_variable_y[i].shape, s_stack.shape, tau_stack.shape)
            full_stack = np.vstack((self.__dual_risk_variable_y[i], s_stack, tau_stack))
            projection = self.__s2_projection_operator[i] @ full_stack
            self.__dual_risk_variable_y[i] = projection[0:self.__dual_risk_variable_y[i].size]
            for k in range(children_of_i.size):
                self.__epigraphical_relaxation_variable_s[stage_at_children_of_i][children_of_i[k]] = \
                    projection[self.__dual_risk_variable_y[i].size + k]
                self.__epigraphical_relaxation_variable_tau[stage_at_children_of_i][children_of_i[k]] = \
                    projection[self.__dual_risk_variable_y[i].size + children_of_i.size + k]

    def proximal_of_f(self):
        # s0 unchanged
        self.project_on_s1()
        self.project_on_s2()
        return "proximal of f complete"

    # operator L and its transpose -------------------------------------------------------------------------------------

    def operator_ell(self):
        for i in range(self.__num_nonleaf_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            self.__dual_part_1_nonleaf[i] = self.__dual_risk_variable_y[i]
            self.__dual_part_2_nonleaf[i] = self.__epigraphical_relaxation_variable_s[stage_at_i][i] \
                - self.__raocp.risk_at_node(i).vector_b.T @ self.__dual_risk_variable_y[i]
            for j in children_of_i:
                self.__dual_part_3_nonleaf[j] = sqrtm(
                    self.__raocp.nonleaf_cost_at_node(j).nonleaf_state_weights) @ self.__states[i]
                self.__dual_part_4_nonleaf[j] = sqrtm(
                    self.__raocp.nonleaf_cost_at_node(j).control_weights) @ self.__controls[i]
                tau = self.__epigraphical_relaxation_variable_tau[stage_at_children_of_i][j]
                self.__dual_part_5_nonleaf[j] = 0.5 * tau
                self.__dual_part_6_nonleaf[j] = 0.5 * tau

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            self.__dual_part_7_leaf[i] = sqrtm(self.__raocp.leaf_cost_at_node(i).leaf_state_weights) \
                @ self.__states[i]
            s = self.__epigraphical_relaxation_variable_s[stage_at_i][i]
            self.__dual_part_8_leaf[i] = 0.5 * s
            self.__dual_part_9_leaf[i] = 0.5 * s

    def operator_ell_transpose(self):
        for i in range(self.__num_nonleaf_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            self.__dual_risk_variable_y[i] = (self.__dual_part_1_nonleaf[i]
                                              - self.__raocp.risk_at_node(i).vector_b
                                              @ self.__dual_part_2_nonleaf[i])\
                .reshape((2 * self.__raocp.tree.children_of(i).size + 1, 1))  # reshape to column vector
            self.__epigraphical_relaxation_variable_s[stage_at_i][i] = self.__dual_part_2_nonleaf[i]
            self.__states[i] = 0
            self.__controls[i] = 0
            for j in children_of_i:
                self.__states[i] += sqrtm(self.__raocp.nonleaf_cost_at_node(j).nonleaf_state_weights).T \
                    @ self.__dual_part_3_nonleaf[j]
                self.__controls[i] += sqrtm(self.__raocp.nonleaf_cost_at_node(j).control_weights).T \
                    @ self.__dual_part_4_nonleaf[j]
                self.__epigraphical_relaxation_variable_tau[stage_at_children_of_i][j] = 0.5 \
                    * (self.__dual_part_5_nonleaf[j] + self.__dual_part_6_nonleaf[j])

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            self.__states[i] = sqrtm(self.__raocp.leaf_cost_at_node(i).leaf_state_weights).T \
                @ self.__dual_part_7_leaf[i]
            self.__epigraphical_relaxation_variable_s[stage_at_i][i] = 0.5 * (self.__dual_part_8_leaf[i]
                                                                              + self.__dual_part_9_leaf[i])

    # proximal of g conjugate ------------------------------------------------------------------------------------------

    def add_halves(self):
        self.__dual_part_5_nonleaf = [j - 0.5 for j in self.__dual_part_5_nonleaf]
        self.__dual_part_6_nonleaf = [j + 0.5 for j in self.__dual_part_6_nonleaf]
        self.__dual_part_8_leaf = [j - 0.5 for j in self.__dual_part_8_leaf]
        self.__dual_part_9_leaf = [j + 0.5 for j in self.__dual_part_9_leaf]

    def subtract_halves(self):
        self.__dual_part_5_nonleaf = [j + 0.5 for j in self.__dual_part_5_nonleaf]
        self.__dual_part_6_nonleaf = [j - 0.5 for j in self.__dual_part_6_nonleaf]
        self.__dual_part_8_leaf = [j + 0.5 for j in self.__dual_part_8_leaf]
        self.__dual_part_9_leaf = [j - 0.5 for j in self.__dual_part_9_leaf]

    def proximal_of_g_conjugate(self):  # not finished
        # precomposition add halves
        self.add_halves()
        # proximal gbar (cone projections)
        # for i in range(self.__num_nonleaf_nodes):
        #     [self.__dual_part_1_nonleaf[i], self.__dual_part_2_nonleaf[i]] = self.__nonleaf_constraint_cone[i]\
        #         .project([self.__dual_part_1_nonleaf[i], self.__dual_part_2_nonleaf[i]])
        #     children_of_i = self.__raocp.tree.children_of(i)
        #     print(np.vstack((self.__dual_part_3_nonleaf[j],
        #                      self.__dual_part_4_nonleaf[j],
        #                      self.__dual_part_5_nonleaf[j],
        #                      self.__dual_part_6_nonleaf[j])))
        #     for j in children_of_i:
        #         projection = self.__nonleaf_second_order_cone[j].project(np.vstack((self.__dual_part_3_nonleaf[j],
        #                                                                             self.__dual_part_4_nonleaf[j],
        #                                                                             self.__dual_part_5_nonleaf[j],
        #                                                                             self.__dual_part_6_nonleaf[j])))

        # prox gbar (.) = proj K K^i K


        # moreau_decomposition
        # prox gast (.) = . - prox g (.)
        pass

    # CHAMBOLLE-POCK ###################################################################################################

    def x_bar(self, copy_x, copy_u, copy_y, copy_s, copy_t):
        # operate L transpose on dual parts
        self.operator_ell_transpose()
        # old primal parts minus (gamma times new primal parts)
        self.__states = [a_i - b_i for a_i, b_i in zip(copy_x, [j * self.__gamma for j in self.__states])]
        self.__controls = [a_i - b_i for a_i, b_i in zip(copy_u, [j * self.__gamma for j in self.__controls])]
        self.__dual_risk_variable_y = [a_i - b_i for a_i, b_i in
                                       zip(copy_y, [j * self.__gamma for j in self.__dual_risk_variable_y])]
        self.__epigraphical_relaxation_variable_s = [a_i - b_i for a_i, b_i in
                                                     zip(copy_s,
                                                         [j * self.__gamma for j in
                                                          self.__epigraphical_relaxation_variable_s])]
        self.__epigraphical_relaxation_variable_tau = [None] + [a_i - b_i for a_i, b_i in
                                                                zip(copy_t[1:],
                                                                    [j * self.__gamma for j in
                                                                    self.__epigraphical_relaxation_variable_tau[1:]])]

    def x_new(self):
        self.proximal_of_f()

    def y_bar(self, copy_x, copy_u, copy_y, copy_s, copy_t,
              copy_w1, copy_w2, copy_w3, copy_w4, copy_w5, copy_w6, copy_w7, copy_w8, copy_w9):
        # modify primal parts
        self.__states = [a_i - b_i for a_i, b_i in zip([j * 2 for j in self.__states], copy_x)]
        self.__controls = [a_i - b_i for a_i, b_i in zip([j * 2 for j in self.__controls], copy_u)]
        self.__dual_risk_variable_y = [a_i - b_i for a_i, b_i in zip([j * 2 for j in self.__dual_risk_variable_y],
                                                                     copy_y)]
        self.__epigraphical_relaxation_variable_s = [a_i - b_i for a_i, b_i in
                                                     zip([j * 2 for j in self.__epigraphical_relaxation_variable_s],
                                                         copy_s)]
        self.__epigraphical_relaxation_variable_tau = [None] + [a_i - b_i for a_i, b_i in
                                                                zip([j * 2 for j in
                                                                     self.__epigraphical_relaxation_variable_tau[1:]],
                                                                    copy_t[1:])]
        # operate L on primal parts
        self.operator_ell()
        # old dual parts plus (gamma times new dual parts)
        self.__dual_part_1_nonleaf = [a_i + b_i for a_i, b_i in
                                      zip(copy_w1, [j * self.__gamma for j in self.__dual_part_1_nonleaf])]
        self.__dual_part_2_nonleaf = [a_i + b_i for a_i, b_i in
                                      zip(copy_w2, [j * self.__gamma for j in self.__dual_part_2_nonleaf])]
        self.__dual_part_3_nonleaf = [a_i + b_i for a_i, b_i in
                                      zip(copy_w3, [j * self.__gamma for j in self.__dual_part_3_nonleaf])]
        self.__dual_part_4_nonleaf = [a_i + b_i for a_i, b_i in
                                      zip(copy_w4, [j * self.__gamma for j in self.__dual_part_4_nonleaf])]
        self.__dual_part_5_nonleaf = [a_i + b_i for a_i, b_i in
                                      zip(copy_w5, [j * self.__gamma for j in self.__dual_part_5_nonleaf])]
        self.__dual_part_6_nonleaf = [a_i + b_i for a_i, b_i in
                                      zip(copy_w6, [j * self.__gamma for j in self.__dual_part_6_nonleaf])]
        self.__dual_part_7_leaf = [a_i + b_i for a_i, b_i in zip(copy_w7, [j * self.__gamma for j in
                                                                 self.__dual_part_7_leaf])]
        self.__dual_part_8_leaf = [a_i + b_i for a_i, b_i in zip(copy_w8, [j * self.__gamma for j in
                                                                 self.__dual_part_8_leaf])]
        self.__dual_part_9_leaf = [a_i + b_i for a_i, b_i in zip(copy_w9, [j * self.__gamma for j in
                                                                 self.__dual_part_9_leaf])]

    def y_new(self):
        self.proximal_of_g_conjugate()

    def chock(self, initial_state, gamma, tolerance=1e-5):
        """
        Chambolle-Pock algorithm
        """
        self.__states[0] = initial_state
        self.__gamma = gamma
        # primal cache
        states_cache = []
        controls_cache = []
        e_cache = []
        keep_running = True
        while keep_running:
            # create copy of parts
            copy_x = self.__states.copy()
            copy_u = self.__controls.copy()
            copy_y = self.__dual_risk_variable_y.copy()
            copy_s = self.__epigraphical_relaxation_variable_s.copy()
            copy_t = self.__epigraphical_relaxation_variable_tau.copy()
            copy_w1 = self.__dual_part_1_nonleaf.copy()
            copy_w2 = self.__dual_part_2_nonleaf.copy()
            copy_w3 = self.__dual_part_3_nonleaf.copy()
            copy_w4 = self.__dual_part_4_nonleaf.copy()
            copy_w5 = self.__dual_part_5_nonleaf.copy()
            copy_w6 = self.__dual_part_6_nonleaf.copy()
            copy_w7 = self.__dual_part_7_leaf.copy()
            copy_w8 = self.__dual_part_8_leaf.copy()
            copy_w9 = self.__dual_part_9_leaf.copy()
            # run primal part of algorithm
            self.x_bar(copy_x, copy_u, copy_y, copy_s, copy_t)
            self.x_new()
            # calculate error
            current_error = max([np.linalg.norm(j, np.inf)
                                 for j in [a_i - b_i for a_i, b_i in zip(self.__states, copy_x)]])
            # create backup copy of primal parts
            new_copy_x = self.__states.copy()
            new_copy_u = self.__controls.copy()
            new_copy_y = self.__dual_risk_variable_y.copy()
            new_copy_s = self.__epigraphical_relaxation_variable_s.copy()
            new_copy_t = self.__epigraphical_relaxation_variable_tau.copy()
            # run dual part of algorithm
            self.y_bar(copy_x, copy_u, copy_y, copy_s, copy_t,
                       copy_w1, copy_w2, copy_w3, copy_w4, copy_w5, copy_w6, copy_w7, copy_w8, copy_w9)
            self.y_new()
            # restore backup copy of primal parts
            self.__states = new_copy_x
            self.__controls = new_copy_u
            self.__dual_risk_variable_y = new_copy_y
            self.__epigraphical_relaxation_variable_s = new_copy_s
            self.__epigraphical_relaxation_variable_tau = new_copy_t
            # add to cache
            states_cache.append(self.__states)
            controls_cache.append(self.__controls)
            # cache error
            e_cache.append(current_error)
            # check stopping criteria
            stopping_criteria = current_error < tolerance
            if stopping_criteria:
                keep_running = False
