import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import time
import raocp.core.cache as cache
import raocp.core.operators as ops
import raocp.core.raocp_spec as spec

import matplotlib.pyplot as plt
import tikzplotlib as tikz


class Bfgs:
    """
    BFGS accelerator
    """

    def __init__(self, memory_size=3, tol=1e-3):
        self.__initial_vector_x = None
        self.__vector_size = self.__initial_vector_x.size
        self.__x_kplus1 = None
        self.__memory = memory_size
        self.__mem_pos_k = None
        self.__mem_pos_kplus1 = None
        self.__mem_pos_oldest = None
        self.__cache_x = [None] * self.__memory
        self.__cache_s = [None] * self.__memory
        self.__cache_y = [None] * self.__memory
        self.__cache_residual = []
        self.__tol = tol
        self.__bfgs_iteration_k = 0
        self.__counter_line_search = None
        self.__counter_zoom = None
        self.__current_direction = None
        self.__max_alpha = 1
        self.__max_line_search_iters = 50
        self.__max_zoom_iters = 50
        self.__alpha_ls = None
        self.__alpha_zoom = None
        self.__zoom_alpha_lo = None
        self.__zoom_alpha_hi = None
        self.__c_1 = 1e-4
        self.__c_2 = 0.9

    def get_first_hessian_approximation(self):
        s_k = self.__cache_s[self.__mem_pos_k]
        y_k = self.__cache_y[self.__mem_pos_k]
        bottom = y_k.T @ y_k
        if bottom <= 1e-16:
            raise Exception("attempted division by zero for H^0_k")
        top = s_k.T @ y_k
        gamma = top / bottom
        h = gamma * np.eye(self.__vector_size)
        if h[0, 0] != h[0, 0]:
            raise Exception("H^0_k element is NaN")
        return h

    def get_direction(self):
        grad = quad_grad(self.__cache_x[self.__mem_pos_k])
        rho = [None] * self.__memory
        grad_bar = [None] * self.__memory
        for i in range():
            bottom = self.__cache_y[i].T @ self.__cache_s[i]
            rho[i] = 1 / bottom

        for i in range():
            grad_bar_bar = self.__cache_s[i].T @ grad
            grad_bar[i] = rho[i] * grad_bar_bar
            grad -= (grad_bar[i] * self.__cache_y[i])

        h_0_k = self.get_first_hessian_approximation()
        r = h_0_k @ grad
        for i in range():
            beta_bar = self.__cache_y[i].T @ r
            beta = rho[i] * beta_bar
            r += (self.__cache_s[i] * (grad_bar[i] - beta))

        self.__current_direction = -r

    def get_step_size_by_line_search(self):
        # Nocedal and Wright, Numerical Optimization, Eq.(3.6)
        x_k_ls = self.__cache_x[self.__mem_pos_k]
        p_k_ls = self.__current_direction
        phi_at_0 = phi(x_k_ls, 0, self.__current_direction)
        phi_dash_at_0 = phi_dash(x_k_ls, 0, self.__current_direction)
        # counter = 0
        self.__counter_line_search = 0
        a_0 = 0
        phi_at_a_i = phi_at_0
        # counter = 1
        self.__counter_line_search = 1
        a_iminus1 = a_0
        a_i = 1
        keep_running = True
        while keep_running:
            phi_at_a_iminus1 = phi_at_a_i
            phi_at_a_i = phi(x_k_ls, a_i, p_k_ls)
            phi_at_0_bar = phi_at_0 + self.__c_1 * a_i * phi_dash_at_0
            if phi_at_a_i > phi_at_0_bar or (phi_at_a_i >= phi_at_a_iminus1 and self.__bfgs_iteration_k > 1):
                self.__zoom_alpha_lo = a_iminus1
                self.__zoom_alpha_hi = a_i
                self.__alpha_ls = self.zoom()
                keep_running = False
            phi_dash_a_i = phi_dash(x_k_ls, a_i, self.__current_direction)
            if abs(phi_dash_a_i) <= -self.__c_2 * phi_dash_at_0 and keep_running:
                self.__alpha_ls = a_i
                keep_running = False
            if phi_dash_a_i >= 0 and keep_running:
                self.__zoom_alpha_lo = a_i
                self.__zoom_alpha_hi = a_iminus1
                self.__alpha_ls = self.zoom()
                keep_running = False
            if keep_running:
                a_iminus1 = a_i
                if self.__counter_line_search == 1:
                    a_i = 0.1
                else:
                    a_i = min(1.1 * a_i, self.__max_alpha)
                self.__counter_line_search = self.__counter_line_search + 1
            if self.__counter_line_search > self.__max_line_search_iters:
                raise Exception(f"alpha line search exceeded {self.__max_line_search_iters} iterations, "
                                f"alpha = {a_i}, bfgs_iter_k = {self.__bfgs_iteration_k}")

    def zoom(self):
        phi_at_a_lo = phi(x_z, self.__zoom_alpha_lo, self.__current_direction)
        self.__counter_zoom = 0
        repeat = True
        while repeat:
            a_j = 0.5 * (self.__zoom_alpha_lo + self.__zoom_alpha_hi)
            phi_at_a_j = phi(x_z, a_j, self.__current_direction)
            phi_0_bar = phi_0 + self.__c_1 * a_j * phi_dash_0
            if phi_at_a_j > phi_0_bar or phi_at_a_j >= phi_at_a_lo:
                a_hi = a_j
            else:
                phi_dash_at_a_j = phi_dash(x_z, a_j, self.__current_direction)
                if abs(phi_dash_at_a_j) <= -self.__c_2 * phi_dash_0:
                    self.__alpha_zoom = a_j
                    repeat = False
                if phi_dash_at_a_j * (self.__zoom_alpha_hi - self.__zoom_alpha_lo) >= 0 and repeat:
                    self.__zoom_alpha_hi = self.__zoom_alpha_lo
                self.__zoom_alpha_lo = a_j
            if repeat:
                self.__counter_zoom = self.__counter_zoom + 1
                if self.__counter_zoom > self.__max_zoom_iters:
                    raise Exception(f"zoom search exceeded {self.__max_zoom_iters} iterations, "
                                    f"alpha = {a_j}, bfgs_iter_k = {self.__bfgs_iteration_k}")

    def cache_initial_x(self):
        self.__mem_pos_k = 0
        self.__mem_pos_kplus1 = self.__mem_pos_k + 1
        self.__mem_pos_oldest = 0
        x_k = self.__initial_vector_x
        x_kplus1 = operator_T(self.__initial_vector_x)
        self.__cache_x[self.__mem_pos_k] = x_k
        self.__cache_x[self.__mem_pos_kplus1] = x_kplus1
        self.__cache_s[self.__mem_pos_k] = x_kplus1 - x_k
        self.__cache_y[self.__mem_pos_k] = grad_k_plus_1 - grad_k
        self.__mem_pos_k = 1
        self.__mem_pos_kplus1 = self.__mem_pos_k + 1

    def cache_x_s_y(self):
        x_k = self.__cache_x[self.__mem_pos_k]
        grad_k_plus_1 = quad_grad(self.__x_kplus1)
        grad_k = quad_grad(x_k)
        self.__cache_x[self.__mem_pos_kplus1] = self.__x_kplus1
        self.__cache_s[self.__mem_pos_k] = self.__x_kplus1 - x_k
        self.__cache_y[self.__mem_pos_k] = grad_k_plus_1 - grad_k
        self.__mem_pos_k = self.__mem_pos_k + 1
        self.__mem_pos_kplus1 = self.__mem_pos_k + 1
        self.__mem_pos_oldest = 0

    def bfgs(self, x_0):
        self.__initial_vector_x = x_0
        grad_dir = quad_grad(x_0)
        alpha = self.get_step_size_by_line_search()
        self.__x_kplus1 = x_0 - alpha * grad_dir
        self.cache_x_s_y()
        self.__bfgs_iteration_k = 1
        keep_running = True
        print("started")
        while keep_running:
            x_k = self.__cache_x[self.__mem_pos_k]
            self.get_direction()
            alpha_k = self.get_step_size_by_line_search()
            self.__x_kplus1 = x_k + alpha_k * self.__current_direction
            self.cache_x_s_y()
            residual = np.linalg.norm(self.__cache_s[self.__mem_pos_k], 2)
            self.__cache_residual.append(residual)
            keep_running = residual > self.__tol and self.__bfgs_iteration_k < 20
            if keep_running:
                self.__bfgs_iteration_k = self.__bfgs_iteration_k + 1

        return self.__x_kplus1
