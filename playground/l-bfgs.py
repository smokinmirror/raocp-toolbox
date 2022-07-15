import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits import mplot3d


'''
quad = 0.5 * x.T*Q*x + q.T*x
'''


# ----
# quad objective
def quad_func(x_):
    f = 0.5 * x_.T@Q@x_ + q.T@x_
    return f


# quad gradient
def quad_grad(x_):
    gf = Q@x_ + q
    return gf


# ----
# one dimensional function of solver method result
def phi(x_, a_, p_):
    if a_ == 0:
        return 0.5 * np.linalg.norm(x_, 2)**2
    else:
        return 0.5 * np.linalg.norm(x_ + a_*p_, 2)**2


# derivative of phi
def phi_dash(x_, a_, p_):
    if a_ == 0:
        dash = x_.T @ p_
        if dash > 0:
            raise Exception("derivative of phi at alpha = 0 is positive")
        return dash
    else:
        return (x_ + a_*p_).T @ p_


########################################################################################################################
def get_first_hessian_approximation():
    s = s_cache[-1]
    y = y_cache[-1]
    bottom = y.T @ y
    if bottom <= 1e-16:
        raise Exception("attempted division by zero for H^0_k")
    top = s.T@y
    gamma = top / bottom
    h = gamma * np.eye(n)
    if h[0, 0] != h[0, 0]:
        raise Exception("H^0_k element is NaN")
    return h


def get_direction(k):
    grad = quad_grad(x_cache[-1])
    rho = [None] * k
    grad_bar = [None] * k
    for i in range(max(k-m, 0), k):
        bottom = y_cache[i].T @ s_cache[i]
        rho[i] = 1 / bottom

    for i in range(k-1, max(k-m-1, -1), -1):
        grad_bar_bar = s_cache[i].T@grad
        grad_bar[i] = rho[i] * grad_bar_bar
        grad -= (grad_bar[i] * y_cache[i])

    h_0_k = get_first_hessian_approximation()
    r = h_0_k@grad
    for i in range(max(k-m, 0), k):
        beta_bar = y_cache[i].T@r
        beta = rho[i] * beta_bar
        r += (s_cache[i] * (grad_bar[i] - beta))

    return -r


def get_step_size_by_line_search(direction_, k):
    # Nocedal and Wright, Numerical Optimization, Eq.(3.6)
    c_1_ls = 1e-4
    c_2_ls = 0.9
    # preliminaries
    x_ls = x_cache[-1]
    p_ls = direction_
    a_max = 1
    phi_at_0 = phi(x_ls, 0, direction_)
    phi_dash_at_0 = phi_dash(x_ls, 0, direction_)
    alpha_ls = None
    max_ls = 50
    # ls_counter = 0
    a_0 = 0
    phi_at_a_i = phi_at_0
    ls_counter = 1
    a_iminus1 = a_0
    a_i = 1
    keep_running = True
    while keep_running:
        phi_at_a_iminus1 = phi_at_a_i
        phi_at_a_i = phi(x_ls, a_i, p_ls)
        phi_at_0_bar = phi_at_0 + c_1_ls * a_i * phi_dash_at_0
        if phi_at_a_i > phi_at_0_bar or (phi_at_a_i >= phi_at_a_iminus1 and k > 1):
            alpha_ls = zoom(a_iminus1, a_i, x_ls, p_ls, c_1_ls, c_2_ls, phi_at_0, phi_dash_at_0, k)
            keep_running = False
        phi_dash_a_i = phi_dash(x_ls, a_i, direction_)
        if abs(phi_dash_a_i) <= -c_2_ls*phi_dash_at_0 and keep_running:
            alpha_ls = a_i
            keep_running = False
        if phi_dash_a_i >= 0 and keep_running:
            alpha_ls = zoom(a_i, a_iminus1, x_ls, p_ls, c_1_ls, c_2_ls, phi_at_0, phi_dash_at_0, k)
            keep_running = False
        if keep_running:
            a_iminus1 = a_i
            if ls_counter == 1:
                a_i = 0.5 * (0.5 + a_max)
            else:
                a_i = min(1.1 * a_i, a_max)
            ls_counter += 1
        if ls_counter > max_ls:
            raise Exception(f"alpha search exceeded {max_ls} iterations, alpha = {a_i}, k = {k}")

    return alpha_ls


def zoom(a_lo, a_hi, x_z, p_z, c_1_z, c_2_z, phi_0, phi_dash_0, k):
    alpha_zoom = None
    phi_at_a_lo = phi(x_z, a_lo, p_z)
    counter_zoom = 0
    max_zoom = 50
    repeat = True
    while repeat:
        a_j = 0.5 * (a_lo + a_hi)
        phi_at_a_j = phi(x_z, a_j, p_z)
        phi_0_bar = phi_0 + c_1_z * a_j * phi_dash_0
        if phi_at_a_j > phi_0_bar or phi_at_a_j >= phi_at_a_lo:
            a_hi = a_j
        else:
            phi_dash_at_a_j = phi_dash(x_z, a_j, p_z)
            if abs(phi_dash_at_a_j) <= -c_2_z*phi_dash_0:
                alpha_zoom = a_j
                repeat = False
            if phi_dash_at_a_j * (a_hi - a_lo) >= 0 and repeat:
                a_hi = a_lo
            a_lo = a_j
        if repeat:
            counter_zoom += 1
            if counter_zoom > max_zoom:
                raise Exception(f"zoom search exceeded {max_zoom} iterations, alpha = {a_j}, k = {k}")

    return alpha_zoom


def get_x_kplus1(k):
    x_k = x_cache[-1]
    p_k = get_direction(k)
    alpha_k = get_step_size_by_line_search(p_k, k)
    if k >= m:
        s_cache[k-m] = None
        y_cache[k - m] = None
    return x_k + alpha_k*p_k


def cache_x_s_y(x_new):
    x_k_plus_1 = x_new
    x_k = x_cache[-1]
    grad_k_plus_1 = quad_grad(x_k_plus_1)
    grad_k = quad_grad(x_k)
    s_cache.append(x_k_plus_1 - x_k)
    y_cache.append(grad_k_plus_1 - grad_k)
    x_cache.append(x_k_plus_1)


########################################################################################################################
# ----
# L-BFGS method
def main_ell_bfgs():
    # quad problem with L-BFGS method
    x_new = None
    k = 0
    grad_dir = quad_grad(x_0)
    alpha = get_step_size_by_line_search(-grad_dir, k)
    x_plus = x_0 - alpha*grad_dir
    cache_x_s_y(x_plus)
    k = 1
    keep_running = True
    residual_cache = []
    print("started")
    while keep_running:
        x_new = get_x_kplus1(k)
        cache_x_s_y(x_new)
        residual = np.linalg.norm(x_cache[-1] - x_cache[-2], 2)
        residual_cache.append(residual)
        keep_running = residual > tol and k < 20
        if keep_running:
            k += 1

    return x_new, residual_cache


# ----
# Specification
tol = 1e-3
n = 1000
m = 5
C = 0.1 * np.random.randn(n, n)
Q = np.eye(n) + C.T@C
q = np.random.randn(n).reshape(-1, 1)
x_0 = np.random.randn(n).reshape(-1, 1)
print(f"Q = \n{np.linalg.norm(np.diagonal(Q), np.inf)}")

# ----
# Cache
x_cache = [x_0]
s_cache = []  # * m
y_cache = []  # * m

# ----
# Solve in cvxpy
x = cp.Variable(n)
cost = 0.5 * cp.quad_form(x, Q) + q.T@x
cp_prob = cp.Problem(cp.Minimize(cost))
cp_prob.solve()

# ----
# Run
solution, cache = main_ell_bfgs()
plt.semilogy(cache)
plt.show()

# ----
# Check
if not np.allclose(solution.T, x.value, atol=1e-3):
    raise Exception("solutions not close")
