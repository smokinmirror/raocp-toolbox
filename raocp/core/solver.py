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
        self.__operator = ops.Operator(self.__raocp, self.__cache.get_primal_split(), self.__cache.get_dual_split())
        self.__primal = None
        self.__old_primal = None
        self.__dual = None
        self.__old_dual = None
        self.__parameter_1 = None
        self.__parameter_2 = None

    def primal_k_plus_half(self):
        dual, _ = self.__cache.get_dual()
        # operate L transpose on dual parts
        ell_transpose_dual = self.__operator.ell_transpose(dual)
        # get old primal
        _, old_primal = self.__cache.get_primal()
        # old primal minus (alpha1 times ell_transpose_dual)
        self.__cache._Cache__primal = [a_i - b_i for a_i, b_i in zip(old_primal, [j * self.__parameter_1
                                                                                  for j in ell_transpose_dual])]

    def primal_k_plus_one(self):
        self.__cache.proximal_of_f()

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

    def chock(self, initial_state, alpha1=1.0, alpha2=1.0, max_iters=10, tol=1e-5):
        """
        Chambolle-Pock algorithm
        """
        self.__primal[0] = initial_state
        self.__parameter_1 = alpha1
        self.__parameter_2 = alpha2
        max_iterations = max_iters
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
            current_error = max([np.linalg.norm(j, np.inf)
                                 for j in [a_i - b_i for a_i, b_i in zip(self.__states, copy_x)]])

            # cache variables
            self.__cache._update_cache()

            # cache error
            error_cache.append(current_error)
            print(current_error)

            # check stopping criteria
            stopping_criteria = current_iteration > max_iterations
            if stopping_criteria:
                keep_running = False
            current_iteration += 1
