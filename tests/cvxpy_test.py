import cvxpy as cp
import numpy as np

# Problem data.
num = 3
initial_state = np.ones((num, 1))
A = np.eye(num)
B = np.eye(num)
# x0, x1, x2, x3, u0, u1, u2
x_and_u_bar = np.array((np.random.randn(num), np.random.randn(num), np.random.randn(num), np.random.randn(num),
                        np.random.randn(num), np.random.randn(num), np.random.randn(num)))
print(x_and_u_bar.shape)

# Construct the problem.
x_and_u = cp.Variable(x_and_u_bar.shape)
objective = cp.Minimize(cp.sum_squares(x_and_u - x_and_u_bar))
constraints = [x_and_u_bar[0] == initial_state]
prob = cp.Problem(objective, constraints)

cost = 0
constr = []
for t in range(4):
    cost += cp.sum_squares(x[:,t+1]) + cp.sum_squares(u[:,t])
    constr += [x[:,t+1] == A@x[:,t] + B@u[:,t],
               cp.norm(u[:,t], 'inf') <= 1]
# sums problem objectives and concatenates constraints.
constr += [x[:,T] == 0, x[:,0] == x_0]
problem = cp.Problem(cp.Minimize(cost), constr)
problem.solve(solver=cp.ECOS)

# # The optimal objective value is returned by `prob.solve()`.
# result = prob.solve()
# # The optimal value for x is stored in `x.value`.
# print(x.value)
# # The optimal Lagrange multiplier for a constraint is stored in
# # `constraint.dual_value`.
# # print(constraints[0].dual_value)
