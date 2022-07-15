import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import time
import raocp.core.cache as cache
import raocp.core.operators as ops
import raocp.core.raocp_spec as spec

import matplotlib.pyplot as plt
import tikzplotlib as tikz


class Solver:
    """
    Solver for RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: spec.RAOCP):
        self.__raocp = problem_spec
        self.__cache = cache.Cache(self.__raocp)
        self.__operator = ops.Operator(self.__cache)
        self.__initial_state = None
        self.__parameter_1 = None
        self.__parameter_2 = None
        self.__error = [np.zeros(1)] * 3
        self.__delta_error = [np.zeros(1)] * 3

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

    def _calculate_chock_errors(self):
        # in this function, p = primal and d = dual
        p_new, p = self.__cache.get_primal()
        d_new, d = self.__cache.get_dual()

        # error 1
        p_minus_p_new = [a_i - b_i for a_i, b_i in zip(p, p_new)]
        p_minus_p_new_over_alpha1 = [a_i / self.__parameter_1 for a_i in p_minus_p_new]
        d_minus_d_new = [a_i - b_i for a_i, b_i in zip(d, d_new)]
        _, ell_transpose_d_minus_d_new = self.__cache.get_primal()  # get memory position
        self.__operator.ell_transpose(d_minus_d_new, ell_transpose_d_minus_d_new)
        xi_1 = [a_i - b_i for a_i, b_i in zip(p_minus_p_new_over_alpha1, ell_transpose_d_minus_d_new)]

        # error 2
        d_minus_d_new_over_alpha2 = [a_i / self.__parameter_2 for a_i in d_minus_d_new]
        p_new_minus_p = [a_i - b_i for a_i, b_i in zip(p_new, p)]
        _, ell_p_new_minus_p = self.__cache.get_dual()  # get memory position
        self.__operator.ell(p_new_minus_p, ell_p_new_minus_p)
        xi_2 = [a_i + b_i for a_i, b_i in zip(d_minus_d_new_over_alpha2, ell_p_new_minus_p)]

        # error 0
        _, ell_transpose_error2 = self.__cache.get_primal()  # get memory position
        self.__operator.ell_transpose(xi_2, ell_transpose_error2)
        xi_0 = [a_i + b_i for a_i, b_i in zip(xi_1, ell_transpose_error2)]

        # delta errors
        delta_1 = p_new_minus_p
        delta_2 = [a_i - b_i for a_i, b_i in zip(d_new, d)]
        _, ell_transpose_delta_2 = self.__cache.get_primal()
        self.__operator.ell_transpose(delta_2, ell_transpose_delta_2)
        delta_0 = [a_i - b_i for a_i, b_i in zip(delta_1, ell_transpose_delta_2)]

        return xi_0, xi_1, xi_2, delta_0, delta_1, delta_2

    def chock(self, initial_state, max_iters=10, tol=1e-5):
        """
        Chambolle-Pock algorithm
        """
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)

        # find alpha_1 and _2
        _, prim = self.__cache.get_primal()
        _, dual = self.__cache.get_dual()
        size_prim = np.vstack(prim).size
        size_dual = np.vstack(dual).size
        ell = LinearOperator(dtype=None, shape=(size_dual, size_prim),
                             matvec=self.__operator.linop_ell)
        ell_transpose = LinearOperator(dtype=None, shape=(size_prim, size_dual),
                                       matvec=self.__operator.linop_ell_transpose)
        ell_transpose_ell = ell_transpose * ell
        eigens, _ = eigs(ell_transpose_ell)
        ell_norm = np.real(max(eigens))
        one_over_norm = 0.999 / ell_norm
        self.__parameter_1 = one_over_norm
        self.__parameter_2 = one_over_norm

        current_iteration = 0
        print("timer started")
        tick = time.perf_counter()
        keep_running = True
        while keep_running:
            # run primal part of algorithm
            self.primal_k_plus_half()
            self.primal_k_plus_one()

            # run dual part of algorithm
            self.dual_k_plus_half()
            self.dual_k_plus_one()

            # calculate error
            xi_0, xi_1, xi_2, delta_0, delta_1, delta_2 = self._calculate_chock_errors()
            xi = [xi_0, xi_1, xi_2]
            delta = [delta_0, delta_1, delta_2]
            for i in range(3):
                inf_norm_xi = [np.linalg.norm(a_i, ord=np.inf) for a_i in xi[i]]
                inf_norm_delta = [np.linalg.norm(a_i, ord=np.inf) for a_i in delta[i]]
                self.__error[i] = np.linalg.norm(inf_norm_xi, np.inf)
                self.__delta_error[i] = np.linalg.norm(inf_norm_delta, np.inf)

            current_error = max(self.__error)

            # cache variables
            self.__cache.update_cache()
            # cache error
            if current_iteration == 0:
                self.__error_cache = np.array(self.__error)
                self.__delta_error_cache = np.array(self.__delta_error)
            else:
                self.__error_cache = np.vstack((self.__error_cache, np.array(self.__error)))
                self.__delta_error_cache = np.vstack((self.__delta_error_cache, np.array(self.__delta_error)))

            # check stopping criteria
            if current_iteration >= max_iters:
                keep_running = False
            if current_error <= tol:
                keep_running = False
            if keep_running is True:
                current_iteration += 1

        tock = time.perf_counter()
        print(f"timer stopped in {tock - tick:0.4f} seconds")

        if current_iteration < max_iters:
            return 0  # converged
        elif current_iteration >= max_iters:
            return 1  # not converged
        else:
            raise Exception("Iteration error in solver")

    def print_states(self):
        primal, _ = self.__cache.get_primal()
        seg_p = self.__cache.get_primal_segments()
        print(f"states =\n")
        for i in range(seg_p[1], seg_p[2]):
            print(f"{primal[i]}\n")

    def print_inputs(self):
        primal, _ = self.__cache.get_primal()
        seg_p = self.__cache.get_primal_segments()
        print(f"inputs =\n")
        for i in range(seg_p[2], seg_p[3]):
            print(f"{primal[i]}\n")

    def plot_residuals(self):
        width = 2
        plt.semilogy(self.__error_cache[:, 0], linewidth=width, linestyle="solid")
        plt.semilogy(self.__error_cache[:, 1], linewidth=width, linestyle="solid")
        plt.semilogy(self.__error_cache[:, 2], linewidth=width, linestyle="solid")
        # plt.semilogy(self.__delta_error_cache[:, 0], linewidth=width, linestyle="dashed")
        # plt.semilogy(self.__delta_error_cache[:, 1], linewidth=width, linestyle="dashed")
        # plt.semilogy(self.__delta_error_cache[:, 2], linewidth=width, linestyle="dashed")
        plt.title("Residual values of Chambolle-Pock algorithm iterations")
        plt.ylabel(r"log(residual value)", fontsize=12)
        plt.xlabel(r"iteration", fontsize=12)
        plt.legend(("xi_0", "xi_1", "xi_2"))  # , "delta_0", "delta_1", "delta_2"))
        tikz.save('4-3-residuals.tex')
        plt.show()

    def plot_solution(self):
        primal, _ = self.__cache.get_primal()
        seg_p = self.__cache.get_primal_segments()
        x = primal[seg_p[1]: seg_p[2]]
        u = primal[seg_p[2]: seg_p[3]]
        state_size = np.size(x[0])
        control_size = np.size(u[0])
        raocp = self.__cache.get_raocp()
        num_nonleaf = raocp.tree.num_nonleaf_nodes
        num_nodes = raocp.tree.num_nodes
        num_leaf = num_nodes - num_nonleaf
        num_stages = raocp.tree.num_stages
        fig, axs = plt.subplots(2, state_size, sharex="all", sharey="row")
        fig.set_size_inches(15, 8)
        fig.set_dpi(80)

        for element in range(state_size):
            for i in range(num_leaf):
                j = raocp.tree.nodes_at_stage(num_stages-1)[i]
                plotter = [[raocp.tree.stage_of(j), x[j][element][0]]]
                while raocp.tree.ancestor_of(j) >= 0:
                    anc_j = raocp.tree.ancestor_of(j)
                    plotter.append([[raocp.tree.stage_of(anc_j), x[anc_j][element][0]]])
                    j = anc_j

                x_plot = np.array(np.vstack(plotter))
                axs[0, element].plot(x_plot[:, 0], x_plot[:, 1])
                axs[0, element].set_title(f"state element, x_{element}(t)")

        for element in range(control_size):
            for i in range(num_leaf):
                j = raocp.tree.ancestor_of(raocp.tree.nodes_at_stage(num_stages-1)[i])
                plotter = [[raocp.tree.stage_of(j), u[j][element][0]]]
                while raocp.tree.ancestor_of(j) >= 0:
                    anc_j = raocp.tree.ancestor_of(j)
                    plotter.append([[raocp.tree.stage_of(anc_j), u[anc_j][element][0]]])
                    j = anc_j

                u_plot = np.array(np.vstack(plotter))
                axs[1, element].plot(u_plot[:, 0], u_plot[:, 1])
                axs[1, element].set_title(f"control element, u_{element}(t)")

        for ax in axs.flat:
            ax.set(xlabel='stage, t', ylabel='value')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        fig.tight_layout()
        tikz.save('python-solution.tex')
        plt.show()

