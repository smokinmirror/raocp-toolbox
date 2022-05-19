import numpy as np
import scipy.optimize
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
        self.__num_leaf_nodes = self.__num_nodes - self.__num_nonleaf_nodes
        self.__num_stages = self.__raocp.tree.num_stages
        self.__state_size = self.__raocp.state_dynamics_at_node(1).shape[1]
        self.__control_size = self.__raocp.control_dynamics_at_node(1).shape[1]
        self.__primal_cache = []
        self.__dual_cache = []

        # create primal list
        self._create_primal()

        # create dual list / parts 3,4,5,6 stored in child nodes for convenience
        self._create_dual()

        # create cones
        self._create_cones()

        # dynamics projection
        self.__P = [np.zeros((self.__state_size, self.__state_size))] * self.__num_nodes
        self.__q = [np.zeros((self.__state_size, 1))] * self.__num_nodes
        self.__K = [np.zeros((self.__state_size, self.__state_size))] * self.__num_nonleaf_nodes
        self.__d = [np.zeros((self.__state_size, 1))] * self.__num_nonleaf_nodes
        self.__inverse_of_modified_control_dynamics = [np.zeros((0, 0))] * self.__num_nonleaf_nodes
        self.__sum_of_dynamics = [np.zeros((0, 0))] * self.__num_nodes  # A+BK

        # kernel projection
        self.__kernel_projection_operator = [np.zeros((0, 0))] * self.__num_nonleaf_nodes

        # populate arrays
        self._offline()

        # update cache with iteration zero
        self._update_cache()

    # GETTERS ##########################################################################################################

    def get_primal(self):
        return self.__primal, self.__old_primal

    def get_primal_split(self):
        return self.__primal_split

    def get_dual(self):
        return self.__dual, self.__old_dual

    def get_dual_split(self):
        return self.__dual_split

    # CREATE ###########################################################################################################

    def _create_primal(self):
        self.__primal_split = [0,
                               self.__num_nodes,
                               self.__num_nodes + self.__num_nonleaf_nodes * 1,
                               self.__num_nodes + self.__num_nonleaf_nodes * 2,
                               self.__num_nodes + self.__num_nonleaf_nodes * 2 + (self.__num_stages + 1),
                               self.__num_nodes + self.__num_nonleaf_nodes * 2 + (self.__num_stages + 1) * 2]
        self.__primal = [np.zeros(1)] * self.__primal_split[-1]
        self.__states = self.__primal[self.__primal_split[0]: self.__primal_split[1]]  # x
        self.__controls = self.__primal[self.__primal_split[1]: self.__primal_split[2]]  # u
        self.__dual_risk_y = self.__primal[self.__primal_split[2]: self.__primal_split[3]]  # y
        self.__relaxation_s = self.__primal[self.__primal_split[3]: self.__primal_split[4]]  # s
        self.__relaxation_tau = self.__primal[self.__primal_split[4]: self.__primal_split[5]]  # tau
        for i in range(self.__num_nonleaf_nodes):
            self.__controls[i] = np.zeros((self.__control_size, 1))
            self.__dual_risk_y[i] = np.zeros((2 * self.__raocp.tree.children_of(i).size + 1, 1))

        for i in range(self.__num_stages + 1):
            largest_node_at_stage = max(self.__raocp.tree.nodes_at_stage(i))
            # store variables in their node number inside the stage vector for s and tau
            self.__relaxation_s[i] = np.zeros((largest_node_at_stage + 1, 1))
            if i > 0:
                self.__relaxation_tau[i] = np.zeros((largest_node_at_stage + 1, 1))

    def _create_dual(self):
        self.__dual_split = [0,
                             self.__num_nonleaf_nodes * 1,
                             self.__num_nonleaf_nodes * 2,
                             self.__num_nonleaf_nodes * 2 + self.__num_nodes * 1,
                             self.__num_nonleaf_nodes * 2 + self.__num_nodes * 2,
                             self.__num_nonleaf_nodes * 2 + self.__num_nodes * 3,
                             self.__num_nonleaf_nodes * 2 + self.__num_nodes * 4,
                             self.__num_nonleaf_nodes * 2 + self.__num_nodes * 5,
                             self.__num_nonleaf_nodes * 2 + self.__num_nodes * 6,
                             self.__num_nonleaf_nodes * 2 + self.__num_nodes * 7]
        self.__dual = [np.zeros(1)] * self.__dual_split[-1]
        self.__dual_1_nonleaf = self.__dual[self.__dual_split[0]: self.__dual_split[1]]
        self.__dual_2_nonleaf = self.__dual[self.__dual_split[1]: self.__dual_split[2]]
        self.__dual_3_nonleaf = self.__dual[self.__dual_split[2]: self.__dual_split[3]]
        self.__dual_4_nonleaf = self.__dual[self.__dual_split[3]: self.__dual_split[4]]
        self.__dual_5_nonleaf = self.__dual[self.__dual_split[4]: self.__dual_split[5]]
        self.__dual_6_nonleaf = self.__dual[self.__dual_split[5]: self.__dual_split[6]]
        self.__dual_7_leaf = self.__dual[self.__dual_split[6]: self.__dual_split[7]]
        self.__dual_8_leaf = self.__dual[self.__dual_split[7]: self.__dual_split[8]]
        self.__dual_9_leaf = self.__dual[self.__dual_split[8]: self.__dual_split[9]]
        for i in range(1, self.__num_nodes):
            self.__dual_3_nonleaf[i] = np.zeros((self.__state_size, 1))
            self.__dual_4_nonleaf[i] = np.zeros((self.__control_size, 1))
            if i >= self.__num_nonleaf_nodes:
                self.__dual_7_leaf[i] = np.zeros((self.__state_size, 1))

        print(self.__dual[644])

    def _create_cones(self):
        self.__nonleaf_constraint_cone = [None] * self.__num_nonleaf_nodes
        self.__nonleaf_second_order_cone = [None] * self.__num_nodes
        self.__leaf_second_order_cone = [None] * self.__num_nodes
        for i in range(self.__num_nodes):
            if i < self.__num_nonleaf_nodes:
                self.__nonleaf_constraint_cone[i] = cones.Cartesian([cones.NonnegativeOrthant(),
                                                                     cones.NonnegativeOrthant()])
            if i > 0:
                self.__nonleaf_second_order_cone[i] = cones.SecondOrderCone()
            if i >= self.__num_nonleaf_nodes:
                self.__leaf_second_order_cone[i] = cones.SecondOrderCone()

    # CACHE ############################################################################################################

    def _update_cache(self):
        """
        Update cache list of primal and dual and update 'old' parts to latest list
        """
        # primal
        self.__primal_cache.append(self.__primal.copy())
        self.__old_primal = self.__primal_cache[-1][:]
        self.__old_states = self.__primal_cache[-1][self.__primal_split[0]: self.__primal_split[1]]  # x
        self.__old_controls = self.__primal_cache[-1][self.__primal_split[1]: self.__primal_split[2]]  # u
        self.__old_dual_risk_y = self.__primal_cache[-1][self.__primal_split[2]: self.__primal_split[3]]  # y
        self.__old_relaxation_s = self.__primal_cache[-1][self.__primal_split[3]: self.__primal_split[4]]  # s
        self.__old_relaxation_tau = self.__primal_cache[-1][self.__primal_split[4]: self.__primal_split[5]]  # tau

        # dual
        self.__dual_cache.append(self.__dual.copy())
        self.__old_dual = self.__dual_cache[-1][:]
        self.__old_dual_1_nonleaf = self.__dual_cache[-1][self.__dual_split[0]: self.__dual_split[1]]
        self.__old_dual_2_nonleaf = self.__dual_cache[-1][self.__dual_split[1]: self.__dual_split[2]]
        self.__old_dual_3_nonleaf = self.__dual_cache[-1][self.__dual_split[2]: self.__dual_split[3]]
        self.__old_dual_4_nonleaf = self.__dual_cache[-1][self.__dual_split[3]: self.__dual_split[4]]
        self.__old_dual_5_nonleaf = self.__dual_cache[-1][self.__dual_split[4]: self.__dual_split[5]]
        self.__old_dual_6_nonleaf = self.__dual_cache[-1][self.__dual_split[5]: self.__dual_split[6]]
        self.__old_dual_7_leaf = self.__dual_cache[-1][self.__dual_split[6]: self.__dual_split[7]]
        self.__old_dual_8_leaf = self.__dual_cache[-1][self.__dual_split[7]: self.__dual_split[8]]
        self.__old_dual_9_leaf = self.__dual_cache[-1][self.__dual_split[8]: self.__dual_split[9]]

    # OFFLINE ##########################################################################################################

    def _offline(self):
        """
        Upon creation of Cache class, calculate pre-computable arrays
        """
        self.offline_projection_dynamics()
        self.offline_projection_kernel()

    @staticmethod
    def inverse_using_cholesky(matrix):
        cholesky_of_matrix = np.linalg.cholesky(matrix)
        inverse_of_cholesky = np.linalg.inv(cholesky_of_matrix)
        inverse_of_matrix = inverse_of_cholesky.T @ inverse_of_cholesky
        return inverse_of_matrix

    def offline_projection_dynamics(self):
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__P[i] = np.eye(self.__state_size)

        for i in reversed(range(self.__num_nonleaf_nodes)):
            sum_for_modified_control_dynamics = 0
            sum_for_k = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_modified_control_dynamics += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                    @ self.__raocp.control_dynamics_at_node(j)
                sum_for_k += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                    @ self.__raocp.state_dynamics_at_node(j)

            self.__inverse_of_modified_control_dynamics[i] = \
                self.inverse_using_cholesky(np.eye(self.__control_size) + sum_for_modified_control_dynamics)
            self.__K[i] = - self.__inverse_of_modified_control_dynamics[i] @ sum_for_k
            sum_for_p = 0
            for j in self.__raocp.tree.children_of(i):
                self.__sum_of_dynamics[j] = self.__raocp.state_dynamics_at_node(j) \
                                    + self.__raocp.control_dynamics_at_node(j) @ self.__K[i]
                sum_for_p += self.__sum_of_dynamics[j].T @ self.__P[j] @ self.__sum_of_dynamics[j]

            self.__P[i] = np.eye(self.__state_size) + self.__K[i].T @ self.__K[i] + sum_for_p

    def offline_projection_kernel(self):
        for i in range(self.__num_nonleaf_nodes):
            eye = np.eye(len(self.__raocp.tree.children_of(i)))
            zeros = np.zeros((self.__raocp.risk_at_node(i).matrix_f.shape[1], eye.shape[0]))
            row1 = np.hstack((self.__raocp.risk_at_node(i).matrix_e.T, -eye, -eye))
            row2 = np.hstack((self.__raocp.risk_at_node(i).matrix_f.T, zeros, zeros))
            s2_space = np.vstack((row1, row2))
            kernel = scipy.linalg.null_space(s2_space)
            pseudoinverse_of_kernel = np.linalg.pinv(kernel)
            self.__kernel_projection_operator[i] = kernel @ pseudoinverse_of_kernel

    # ONLINE ###########################################################################################################

    # proximal of f ----------------------------------------------------------------------------------------------------

    def proximal_of_f(self):
        self.proximal_of_relaxation_s_at_stage_zero()
        self.project_on_s1()
        self.project_on_s2()

    def proximal_of_relaxation_s_at_stage_zero(self):
        """
        proximal operator of the identity
        """
        self.__relaxation_s[0] -= 1

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
        print(self.__states[0])
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
            s_stack = self.__relaxation_s[stage_at_children_of_i][children_of_i[0]]
            tau_stack = self.__relaxation_tau[stage_at_children_of_i][children_of_i[0]]
            if children_of_i.size > 1:
                for j in np.delete(children_of_i, 0):
                    s_stack = np.vstack((s_stack,
                                         self.__relaxation_s[stage_at_children_of_i][j]))
                    tau_stack = np.vstack((tau_stack,
                                           self.__relaxation_tau[stage_at_children_of_i][j]))

            full_stack = np.vstack((self.__dual_risk_y[i], s_stack, tau_stack))
            projection = self.__kernel_projection_operator[i] @ full_stack
            self.__dual_risk_y[i] = projection[0:self.__dual_risk_y[i].size]
            for k in range(children_of_i.size):
                self.__relaxation_s[stage_at_children_of_i][children_of_i[k]] = \
                    projection[self.__dual_risk_y[i].size + k]
                self.__relaxation_tau[stage_at_children_of_i][children_of_i[k]] = \
                    projection[self.__dual_risk_y[i].size + children_of_i.size + k]

    # proximal of g conjugate ------------------------------------------------------------------------------------------

    def add_halves(self):
        self.__dual_5_nonleaf = [j - 0.5 for j in self.__dual_5_nonleaf]
        self.__dual_6_nonleaf = [j + 0.5 for j in self.__dual_6_nonleaf]
        self.__dual_8_leaf = [j - 0.5 for j in self.__dual_8_leaf]
        self.__dual_9_leaf = [j + 0.5 for j in self.__dual_9_leaf]

    def subtract_halves(self):
        self.__dual_5_nonleaf = [j + 0.5 for j in self.__dual_5_nonleaf]
        self.__dual_6_nonleaf = [j - 0.5 for j in self.__dual_6_nonleaf]
        self.__dual_8_leaf = [j + 0.5 for j in self.__dual_8_leaf]
        self.__dual_9_leaf = [j - 0.5 for j in self.__dual_9_leaf]

    def proximal_of_g_conjugate(self):  # not finished
        # create copy of dual
        copy_dual = self.__dual.copy()
        # precomposition add halves
        self.add_halves()
        # proximal gbar (cone projections)
        for i in range(self.__num_nonleaf_nodes):
            [self.__dual_1_nonleaf[i], self.__dual_2_nonleaf[i]] = self.__nonleaf_constraint_cone[i]\
                .project([self.__dual_1_nonleaf[i], self.__dual_2_nonleaf[i]])
            children_of_i = self.__raocp.tree.children_of(i)
            for j in children_of_i:
                self.__dual[self.__num_nodes * 2: self.__num_nodes * 6][j] = self.__nonleaf_second_order_cone[j]\
                    .project(self.__dual[self.__num_nodes * 2: self.__num_nodes * 6][j])

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__dual[self.__num_nodes * 6: self.__num_nodes * 9][i] = self.__leaf_second_order_cone[i]\
                .project(self.__dual[self.__num_nodes * 6: self.__num_nodes * 9][i])
        # precomposition subtract halves
        self.subtract_halves()
        # Moreau decomposition
        self.__dual = [a_i - b_i for a_i, b_i in zip(copy_dual, self.__dual)]
