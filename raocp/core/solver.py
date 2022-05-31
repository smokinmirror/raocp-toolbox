import numpy as np
import raocp.core.problem_spec as ps
import raocp.core.cache as cache
import raocp.core.operators as ops
import matplotlib.pyplot as plt


class Solver:
    """
    Solver for RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: ps.RAOCP):
        self.__raocp = problem_spec
        self.__cache = cache.Cache(self.__raocp)
        self.__operator = ops.Operator(self.__cache)
        self.__initial_state = None
        self.__parameter_1 = None
        self.__parameter_2 = None
        self.__xi = [None] * 3
        self.__error = [np.zeros(1)] * 3

    def primal_k_plus_half(self):
        # get memory space for ell_transpose_dual
        _, ell_transpose_dual = self.__cache.get_primal()
        # get current dual
        _, old_dual = self.__cache.get_dual()
        # operate L transpose on dual and store in ell_transpose_dual
        self.__operator.ell_transpose(old_dual, ell_transpose_dual)
        # get old primal
        _, old_primal = self.__cache.get_primal()
        # old primal minus (alpha1 times ell_transpose_dual)
        new_primal = [a_i - b_i for a_i, b_i in zip(old_primal, [j * self.__parameter_1
                                                                 for j in ell_transpose_dual])]
        self.__cache.set_primal(new_primal)

    def primal_k_plus_one(self):
        self.__cache.proximal_of_f(self.__parameter_1)

    def dual_k_plus_half(self):
        # get memory space for ell_transpose_dual
        _, ell_primal = self.__cache.get_dual()
        # get primal k+1 and k
        primal, old_primal = self.__cache.get_primal()
        # two times new primal minus old primal
        modified_primal = [a_i - b_i for a_i, b_i in zip([j * 2 for j in primal], old_primal)]
        # operate L on modified primal
        self.__operator.ell(modified_primal, ell_primal)
        # get old dual
        _, old_dual = self.__cache.get_dual()
        # old dual plus (gamma times ell_primal)
        new_dual = [a_i + b_i for a_i, b_i in zip(old_dual, [j * self.__parameter_2
                                                             for j in ell_primal])]
        self.__cache.set_dual(new_dual)

    def dual_k_plus_one(self):
        self.__cache.proximal_of_g_conjugate(self.__parameter_2)

    def _calculate_errors(self):
        # in this function, p = primal and d = dual
        p_new, p = self.__cache.get_primal()
        d_new, d = self.__cache.get_dual()

        # error 1
        p_minus_p_new = [a_i - b_i for a_i, b_i in zip(p, p_new)]
        p_minus_p_new_over_alpha1 = [a_i / self.__parameter_1 for a_i in p_minus_p_new]
        d_minus_d_new = [a_i - b_i for a_i, b_i in zip(d, d_new)]
        _, ell_transpose_d_minus_d_new = self.__cache.get_primal()  # get memory position
        self.__operator.ell_transpose(d_minus_d_new, ell_transpose_d_minus_d_new)
        self.__xi[1] = [a_i - b_i for a_i, b_i in zip(p_minus_p_new_over_alpha1, ell_transpose_d_minus_d_new)]

        # error 2
        d_minus_d_new_over_alpha2 = [a_i / self.__parameter_2 for a_i in d_minus_d_new]
        p_new_minus_p = [a_i - b_i for a_i, b_i in zip(p_new, p)]
        _, ell_p_new_minus_p = self.__cache.get_dual()  # get memory position
        self.__operator.ell(p_new_minus_p, ell_p_new_minus_p)
        self.__xi[2] = [a_i + b_i for a_i, b_i in zip(d_minus_d_new_over_alpha2, ell_p_new_minus_p)]

        # error 0
        _, ell_transpose_error2 = self.__cache.get_primal()  # get memory position
        self.__operator.ell_transpose(self.__xi[2], ell_transpose_error2)
        self.__xi[0] = [a_i + b_i for a_i, b_i in zip(self.__xi[1], ell_transpose_error2)]

    def chock(self, initial_state, alpha1=1.0, alpha2=1.0, max_iters=10, tol=1e-5):
        """
        Chambolle-Pock algorithm
        """
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)
        self.__parameter_1 = alpha1
        self.__parameter_2 = alpha2
        current_iteration = 0

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
            for i in range(3):
                inf_norm = [np.linalg.norm(a_i, ord=np.inf) for a_i in self.__xi[i]]
                self.__error[i] = np.linalg.norm(inf_norm, np.inf)

            current_error = max(self.__error)

            # cache variables
            self.__cache.update_cache()

            # cache error
            if current_iteration == 0:
                error_cache = np.array(self.__error)
            else:
                error_cache = np.vstack((error_cache, np.array(self.__error)))

            # check stopping criteria
            if current_iteration >= max_iters:
                keep_running = False
            if current_error <= tol:
                keep_running = False
            if keep_running is True:
                current_iteration += 1

        # primal, _ = self.__cache.get_primal()
        # seg_p = self.__cache.get_primal_segments()
        # x = primal[seg_p[1]: seg_p[2]]
        # u = primal[seg_p[2]: seg_p[3]]
        # print(f"x =\n"
        #       f"{x}\n"
        #       f"u =\n"
        #       f"{u}\n")
        width = 3
        plt.semilogy(error_cache[:, 0], linewidth=width, linestyle="solid")
        plt.semilogy(error_cache[:, 1], linewidth=width, linestyle=(0, (5, 10)))
        plt.semilogy(error_cache[:, 2], linewidth=width, linestyle="dashed")
        plt.ylabel(r"residual value", fontsize=12)
        plt.xlabel(r"iteration", fontsize=12)
        plt.legend(("xi_0", "xi_1", "xi_2"))
        plt.show()

        print(np.allclose(error_cache[:, 0], error_cache[:, 1]))
