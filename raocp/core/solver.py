import numpy as np
import raocp.core.problem_spec as ps
import raocp.core.cache as cache
import raocp.core.operators as ops


class Solver:
    """
    Solver for RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: ps.RAOCP):
        self.__raocp = problem_spec
        self.__cache = cache.Cache(self.__raocp)
        _, initial_primal = self.__cache.get_primal()
        _, initial_dual = self.__cache.get_dual()
        self.__operator = ops.Operator(self.__raocp,
                                       initial_primal, self.__cache.get_primal_split(),
                                       initial_dual, self.__cache.get_dual_split())
        self.__initial_state = None
        self.__parameter_1 = None
        self.__parameter_2 = None
        self.__error_0 = None
        self.__error_1 = None
        self.__error_2 = None

    def primal_k_plus_half(self):
        _, old_dual = self.__cache.get_dual()
        # operate L transpose on dual parts
        ell_transpose_dual = self.__operator.ell_transpose(old_dual)
        # get old primal
        _, old_primal = self.__cache.get_primal()
        # old primal minus (alpha1 times ell_transpose_dual)
        self.__cache._Cache__primal = [a_i - b_i for a_i, b_i in zip(old_primal, [j * self.__parameter_1
                                                                                  for j in ell_transpose_dual])]

    def primal_k_plus_one(self):
        self.__cache.proximal_of_f(self.__initial_state, self.__parameter_1)

    def dual_k_plus_half(self):
        # get primal k+1 and k
        primal, old_primal = self.__cache.get_primal()
        # two times new primal minus old primal
        modified_primal = [a_i - b_i for a_i, b_i in zip([j * 2 for j in primal], old_primal)]
        # operate L on modified primal
        ell_primal = self.__operator.ell(modified_primal)
        # get old dual
        _, old_dual = self.__cache.get_dual()
        # old dual plus (gamma times ell_primal)
        self.__cache._Cache__dual = [a_i + b_i for a_i, b_i in zip(old_dual, [j * self.__parameter_2
                                                                              for j in ell_primal])]

    def dual_k_plus_one(self):
        self.__cache.proximal_of_g_conjugate()

    def _calculate_errors(self):
        # in this function, p = primal and d = dual
        p_new, p = self.__cache.get_primal()
        d_new, d = self.__cache.get_dual()
        # error 1
        p_minus_p_new = [a_i - b_i for a_i, b_i in zip(p, p_new)]
        p_minus_p_new_over_alpha1 = [a_i / self.__parameter_1 for a_i in p_minus_p_new]
        d_minus_d_new = [a_i - b_i for a_i, b_i in zip(d, d_new)]
        ell_transpose_d_minus_d_new = self.__operator.ell_transpose(d_minus_d_new)
        self.__error_1 = [a_i - b_i for a_i, b_i in zip(p_minus_p_new_over_alpha1, ell_transpose_d_minus_d_new)]
        # error 2
        d_minus_d_new_over_alpha2 = [a_i / self.__parameter_2 for a_i in d_minus_d_new]
        p_new_minus_p = [a_i - b_i for a_i, b_i in zip(p_new, p)]
        ell_p_new_minus_p = self.__operator.ell(p_new_minus_p)
        self.__error_2 = [a_i + b_i for a_i, b_i in zip(d_minus_d_new_over_alpha2, ell_p_new_minus_p)]
        # error 0
        ell_error2 = self.__operator.ell_transpose(self.__error_2)
        self.__error_0 = [a_i + b_i for a_i, b_i in zip(self.__error_1, ell_error2)]

    def chock(self, initial_state, alpha1=1.0, alpha2=1.0, max_iters=10, tol=1e-5):
        """
        Chambolle-Pock algorithm
        """
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)
        self.__parameter_1 = alpha1
        self.__parameter_2 = alpha2
        current_iteration = 0
        error_cache = []

        keep_running = True
        while keep_running:
            # run primal part of algorithm
            self.primal_k_plus_half()
            self.primal_k_plus_one()

            # run dual part of algorithm
            self.dual_k_plus_half()
            self.dual_k_plus_one()

            # calculate error
            self._calculate_errors()
            self.__error_0 = [np.linalg.norm(a_i, ord=2) for a_i in self.__error_0]
            self.__error_1 = [np.linalg.norm(a_i, ord=2) for a_i in self.__error_1]
            self.__error_2 = [np.linalg.norm(a_i, ord=2) for a_i in self.__error_2]
            current_error = max([np.linalg.norm(self.__error_0, np.inf),
                                 np.linalg.norm(self.__error_1, np.inf),
                                 np.linalg.norm(self.__error_2, np.inf)])

            # cache variables
            self.__cache.update_cache()

            # cache error
            print(current_error)
            error_cache.append(current_error)

            # check stopping criteria
            if current_iteration >= max_iters:
                keep_running = False
            if current_error <= tol:
                keep_running = False
            if keep_running is True:
                current_iteration += 1
