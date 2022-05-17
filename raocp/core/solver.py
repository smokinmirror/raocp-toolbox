import numpy as np
import raocp.core.problem_spec as ps
import raocp.core.cache as cache


class Solver:
    """
    Solver for RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: ps.RAOCP):
        self.__raocp = problem_spec
        self.__solver_parameter_1 = None
        self.__solver_parameter_2 = None
        self.__create_cache()

    def __create_cache(self):
        self.__cache = cache.Cache(self.__raocp)

    def primal_k_plus_half(self, copy_x, copy_u, copy_y, copy_s, copy_t):
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

    def primal_k_plus_one(self):
        self.proximal_of_f()

    def dual_k_plus_half(self, copy_x, copy_u, copy_y, copy_s, copy_t,
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

    def dual_k_plus_one(self):
        self.proximal_of_g_conjugate()

    def chock(self, initial_state, alpha1=1.0, alpha2=1.0, max_iters=10, tol=1e-5):
        """
        Chambolle-Pock algorithm
        """
        # self.__states[0] = initial_state
        self.__solver_parameter_1 = alpha1
        self.__solver_parameter_2 = alpha2
        max_iterations = max_iters
        current_iteration = 0

        # primal cache
        states_cache = []
        controls_cache = []
        e_cache = []

        keep_running = True
        while keep_running:
            # run primal part of algorithm
            self.primal_k_plus_half()
            self.primal_k_plus_one()
            # calculate error
            current_error = max([np.linalg.norm(j, np.inf)
                                 for j in [a_i - b_i for a_i, b_i in zip(self.__states, copy_x)]])

            # run dual part of algorithm
            self.dual_k_plus_half()
            self.dual_k_plus_one()

            # cache variables
            states_cache.append(self.__states)
            controls_cache.append(self.__controls)
            # cache error
            e_cache.append(current_error)
            print(current_error)

            # check stopping criteria
            stopping_criteria = current_iteration > max_iterations
            if stopping_criteria:
                keep_running = False
            current_iteration += 1
