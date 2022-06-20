import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


'''
quad = problem of (0.5*xT*Q*x + qT*x + c)

square = problem of 0.5*||A*x - b||^2 + gamma*||x||_1
'''


# ----
# Problem one
def f_and_gf_quad(x_):
    f = 0.5 * x_.T@Q@x_ + q.T@x_ + c
    gf = Q@x_ + q
    return f, gf


# ----
# Problem three
def f_and_gf_square(x_):
    r = A@x_ - b
    f = 0.5 * (r.T@r) + gamma * sum(np.abs(x_))
    gf = A.T @ r + gamma
    gf_square_only = A.T@r
    return f, gf, gf_square_only


# ----
def step_size_quad():
    L = np.linalg.norm(Q, 2)
    beta_ = 0.9 / L
    return beta_


# ----
def step_size_square():
    L = np.linalg.norm(A.T@A, 2)
    beta_ = 0.9 / L
    return beta_


# ----
def projection_onto_Rplus(v):
    for i in range(v.size):
        v[i] = max(v[i], 0)

    return v


# ----
def soft_thresholding(u, t):
    for j in range(u.size):
        if u[j] < -t:
            u[j] += t
        elif u[j] > t:
            u[j] -= t
        else:
            u[j] = 0

    return u


# ----
# Gradient descent method
def main_gradient_descent():
    # Problem one with gradient descent method
    x_old = x_guess
    x_cache_one = x_guess
    beta = step_size_quad()
    keep_running = True
    e_cache_one = []
    while keep_running:
        _, gf = f_and_gf_quad(x_old)
        x_new = x_old - beta*gf
        x_cache_one = np.vstack((x_cache_one, x_new))
        current_error = np.linalg.norm(x_new - x_old, np.inf)
        e_cache_one += [current_error]
        keep_running = current_error > e
        x_old = x_new

    return x_cache_one, e_cache_one


# ----
# Projected gradient method
def main_projected_gradient():
    # Problem two with projected gradient method
    x_old = x_guess
    x_cache_two = x_guess
    beta = step_size_quad()
    keep_running = True
    e_cache_two = []
    while keep_running:
        _, gf = f_and_gf_quad(x_old)
        x_new = projection_onto_Rplus(x_old - beta*gf)
        x_cache_two = np.vstack((x_cache_two, x_new))
        current_error = np.linalg.norm(x_new - x_old, np.inf)
        e_cache_two += [current_error]
        keep_running = current_error > e
        x_old = x_new

    return x_cache_two, e_cache_two


# ----
# Proximal gradient method
def main_proximal_gradient():
    # Problem three with proximal gradient method
    x_old = x_guess
    x_cache_three = x_guess
    beta = step_size_square()
    keep_running = True
    e_cache_three = []
    while keep_running:
        _, _, gf = f_and_gf_square(x_old)
        x_new = soft_thresholding(x_old - beta * gf, gamma * beta)
        x_cache_three = np.vstack((x_cache_three, x_new))
        current_error = np.linalg.norm(x_new - x_old, np.inf)
        e_cache_three += [current_error]
        keep_running = current_error > e
        x_old = x_new

    return x_cache_three, e_cache_three


# ----
# Alternating direction method of multipliers
def main_admm():
    # Problem four with admm
    rho = 1  # scaling term - must be > 0
    x_old = x_guess
    z_old = np.zeros(n)
    u_old = np.zeros(n)
    x_cache_four = x_old
    z_cache_four = z_old
    u_cache_four = u_old
    e_cache_four = []
    keep_running = True
    while keep_running:
        # iterate over
        inv_term = A.T@A + rho*np.eye(n)  # always invertible because rho > 0
        x_new = np.linalg.solve(inv_term, (A.T@b + rho*(z_old - u_old)))  # avoid .inv
        z_new = soft_thresholding(x_new + u_old, gamma / rho)
        u_new = u_old + x_new - z_new
        # cache
        x_cache_four = np.vstack((x_cache_four, x_new))
        z_cache_four = np.vstack((z_cache_four, z_new))
        u_cache_four = np.vstack((u_cache_four, u_new))
        current_error = max(np.linalg.norm(x_new - x_old, np.inf),
                            np.linalg.norm(z_new - z_old, np.inf),
                            np.linalg.norm(x_new - z_new, np.inf))
        e_cache_four += [current_error]
        keep_running = current_error > e
        # update variables
        x_old = x_new
        z_old = z_new
        u_old = u_new
        rho *= 1.1  # iteratively increase rho
        rho = min(20, rho)  # limit increase of rho
    return x_cache_four, e_cache_four, z_cache_four, u_cache_four, rho

# ----
# Chambolle-Pock method
def main_chambolle_pock():
    # Problem five with Chambolle-Pock algorithm
    alpha = 5  # scaling term - must be 0 < a < 1
    x_old = x_guess
    u_old = np.zeros(n)
    x_cache_five = x_old
    u_cache_five = u_old
    e_cache_five_x = []
    e_cache_five_u = []
    keep_running = True
    while keep_running:
        # x_k+1
        v1 = x_old - alpha*u_old
        x_new = np.linalg.solve(A.T@A + (1/alpha)*np.eye(n), v1/alpha + A.T@b)
        # u_k+1
        v2 = u_old + alpha*(2*x_new - x_old)
        u_new = v2 - alpha*soft_thresholding(v2/alpha, gamma/alpha)  # using Moreau's identity
        # cache
        x_cache_five = np.vstack((x_cache_five, x_new))
        u_cache_five = np.vstack((u_cache_five, u_new))
        current_error_x = np.linalg.norm(x_old - x_new, np.inf)
        current_error_u = np.linalg.norm(u_old - u_new, np.inf)
        e_cache_five_x += [current_error_x]
        e_cache_five_u += [current_error_u]
        keep_running = max(current_error_x > e, current_error_u > e)
        # update variables
        x_old = x_new
        u_old = u_new

    return x_cache_five, e_cache_five_x, e_cache_five_u


# ----
# Parameter Setup
x_guess = np.array([0, 0])
Q = np.diagflat([20, 8])
A = Q
q = np.array([2, -8])
b = q
c = 20
gamma = c
e = 1e-13
n = np.shape(Q)[0]

# ----
# Problem selection
prob = 5

match prob:
    # Problem one
    case 1: x_cache, e_cache = main_gradient_descent()

    # Problem two
    case 2: x_cache, e_cache = main_projected_gradient()

    # Problem three
    case 3: x_cache, e_cache = main_proximal_gradient()

    # Problem three
    case 4:
        x_cache, e_cache, z_cache, u_cache, rho_ = main_admm()
        x_last = x_cache[-1, :].copy()
        z_last = z_cache[-1, :].copy()
        u_last = u_cache[-1, :].copy()
        x_check = np.linalg.solve(A.T@A + rho_*np.eye(n), A.T@b + rho_*(z_last - u_last))
        z_check = soft_thresholding(x_last + u_last, gamma / rho_)
        print("x = 0 ? :", x_check - x_cache[-1, :])
        print("z = 0 ? :", z_check - z_cache[-1, :])

    # Problem three
    case 5: x_cache, e_cache_x, e_cache_u = main_chambolle_pock()


# plot e
plt.plot(e_cache_x)
plt.plot(e_cache_u)
plt.legend(("primal", "dual"))
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("suboptimality (error)")
plt.show()

# # plot x
# plt.plot(x_cache)
# plt.legend(("x1", "x2"))
# plt.show()

# # plot x against f
# rows = np.shape(x_cache)[0]
# f_min = np.zeros(rows)
# if prob <= 2:
#     for k in range(rows):
#         f_min[k], _, __ = f_and_gf_square(x_cache[k, :])
#
#     print(f_min[-1])
# else:
#     for k in range(rows):
#         f_min[k], _, __ = f_and_gf_square(x_cache[k, :])
#
#     print(f_min[-1])
# ax = plt.axes(projection='3d')
# ax.plot3D(x_cache[:, 0], x_cache[:, 1], f_min, 'blue')
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('f')
# plt.show()