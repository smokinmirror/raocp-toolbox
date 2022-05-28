import cvxpy as cp
import numpy as np

# Problem data.
num = 3
initial_state = np.ones((num,))
A = np.eye(num) + np.vstack((np.random.randn(num), np.random.randn(num), np.random.randn(num)))
B = np.eye(num)
# x0, x1, x2, x3, u0, u1, u2
x_bar = np.array((initial_state, np.random.randn(num), np.random.randn(num), np.random.randn(num)))
u_bar = np.array((np.random.randn(num), np.random.randn(num), np.random.randn(num)))

# Construct the problem.
T = x_bar.shape[0] - 1
x = cp.Variable(x_bar.shape)
u = cp.Variable(u_bar.shape)
# sums problem objectives and concatenates constraints.
cost = 0
constraints = []
for t in range(T - 1):
    cost += cp.sum_squares(x[t, :] - x_bar[t, :]) + cp.sum_squares(u[t, :] - u_bar[t, :])
    constraints += [x[t + 1, :] == A @ x[t, :] + B @ u[t, :]]

constraints += [x[0, :] == initial_state]
cost += cp.sum_squares(x[T, :] - x_bar[T, :])
objective = cp.Minimize(cost)
problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.ECOS)

# The optimal value for x is stored in `x.value`.
print(f"optimal x arg = {x.value}\n"
      f"optimal u arg = {u.value}")
