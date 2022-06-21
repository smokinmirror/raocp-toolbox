import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


'''
quad = 0.5 * x.T*Q*x + q.T*x

square = 0.5 * ||A*x - b||^2 + gamma*||x||_1
'''


# ----
# quad problem
def f_and_gf_quad(x_):
    f = 0.5 * x_.T@Q@x_ + q.T@x_
    gf = Q@x_ + q
    return f, gf


# # ----
# # square problem
# def f_and_gf_square(x_):
#     r = A@x_ - b
#     f = 0.5 * (r.T@r) + gamma * sum(np.abs(x_))
#     gf = A.T @ r + gamma
#     gf_square_only = A.T@r
#     return f, gf, gf_square_only


########################################################################################################################
def get_h_0_k():
    s = s_cache[-1]
    y = y_cache[-1]
    if y.T@y <= 1e-16:
        raise Exception("attempted division by zero for H^0_k")
    gamma = s.T@y / y.T@y
    h = gamma * np.eye(n)
    if h[0, 0] != h[0, 0]:
        raise Exception("H^0_k element is NaN")
    return h


def get_p_k(k):
    _, grad = problem(x_cache[-1])
    rho = [None] * k
    grad_bar = [None] * k
    for i in range(max(k-m, 0), k):
        rho[i] = 1 / (y_cache[i].T @ s_cache[i])

    for i in range(k-1, max(k-m-1, -1), -1):
        grad_bar[i] = rho[i] * (s_cache[i].T@grad)
        grad -= (grad_bar[i] * y_cache[i])

    h_0_k = get_h_0_k()
    r = h_0_k@grad
    for i in range(max(k-m, 0), k):
        beta = rho[i] * y_cache[i].T@r
        r += s_cache[i] * (grad_bar[i] - beta)

    return -r


def get_alpha_k(p, k):
    # Nocedal and Wright, Numerical Optimization, Eq.(3.6)
    c_1 = 1e-4
    c_2 = 0.9
    alpha = 1
    x_k = x_cache[-1]
    f_x, grad_f_x = problem(x_k)
    keep_running = True
    counter = 0
    while keep_running:
        f_x_plus_alpha_p, grad_f_x_plus_alpha_p = problem(x_k + alpha*p)
        sufficient_decrease_cond = f_x + c_1*alpha*grad_f_x.T@p - f_x_plus_alpha_p
        curvature_cond = grad_f_x_plus_alpha_p.T@p - c_2*grad_f_x.T@p
        stop = sufficient_decrease_cond >= 0 and curvature_cond >= 0
        if stop:
            keep_running = False
        else:
            alpha *= 0.5
            counter += 1
        if counter > 50:
            raise Exception(f"alpha search exceeded 50 iterations, alpha = {alpha}, k = {k}")

    return alpha


def get_x_plus_1(k):
    x_k = x_cache[-1]
    p_k = get_p_k(k)
    alpha_k = get_alpha_k(p_k, k)
    if k >= m:
        s_cache[k-m] = None
        y_cache[k - m] = None
    return x_k + alpha_k*p_k


def cache_x_s_y(x_new):
    x_k_plus_1 = x_new
    x_k = x_cache[-1]
    _, grad_k_plus_1 = problem(x_k_plus_1)
    _, grad_k = problem(x_k)
    s_cache.append(x_k_plus_1 - x_k)
    y_cache.append(grad_k_plus_1 - grad_k)
    x_cache.append(x_k_plus_1)


########################################################################################################################
# ----
# L-BFGS method
def main_ell_bfgs():
    # quad problem with L-BFGS method#
    k = 0
    _, grad_dir = problem(x_0)
    alpha = get_alpha_k(-grad_dir, k)
    x_plus = x_0 - alpha*grad_dir
    cache_x_s_y(x_plus)
    k = 1
    keep_running = True
    grad_cache = []
    print("started")
    while keep_running:
        x_new = get_x_plus_1(k)
        cache_x_s_y(x_new)
        _, grad = problem(x_new)
        max_grad = np.linalg.norm(grad, np.inf)
        keep_running = max_grad > tol and k < 100
        grad_cache.append(max_grad)
        if keep_running:
            k += 1

    return x_new, grad_cache


# ----
# Specification
tol = 1e-3
n = 1000
m = 5
problem = f_and_gf_quad
C = 0.1 * np.random.randn(n, n)
Q = np.eye(n) + C.T@C
q = np.random.randn(n).reshape(-1, 1)
x_0 = np.random.randn(n).reshape(-1, 1)
print(f"Q = \n{Q}")
# print(f"q = \n{q}")
# print(f"x_0 = \n{x_0}")

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
solution, cache_grad = main_ell_bfgs()
if not np.allclose(solution.T, x.value, atol=tol):
    raise Exception("solutions not close")

plt.semilogy(cache_grad)
plt.show()
