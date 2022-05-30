import raocp as r
import numpy as np
import scipy as sp

# ScenarioTree generation ----------------------------------------------------------------------------------------------

p = np.array([[0.1, 0.8, 0.1],
              [0.4, 0.6, 0],
              [0, 0.3, 0.7]])

v = np.array([0.5, 0.4, 0.1])

(N, tau) = (2, 1)
tree = r.core.MarkovChainScenarioTreeFactory(transition_prob=p,
                                             initial_distribution=v,
                                             num_stages=N, stopping_time=tau).create()

tree.bulls_eye_plot(dot_size=5, radius=300)

tree.set_data_at_node(4, {"a": 1})
print(sum(tree.probability_of_node(tree.nodes_at_stage(8))))

print(tree)

# RAOCP generation -----------------------------------------------------------------------------------------------------

# Aw = 0.5 * np.eye(3)
# As = [Aw, 2 * Aw, 3 * Aw]  # n x n matrices
#
# Bw = 1.5 * np.eye(3)
# Bs = [Bw, 2 * Bw, 3 * Bw]  # n x u matrices
#
# cost_type = "Quadratic"
# cost_types = [cost_type] * 3
# Q = 10 * np.eye(3)  # n x n matrix
# Qs = [Q, 2 * Q, 3 * Q]
# R = np.eye(3)  # u x u matrix OR scalar
# Rs = [R, 2 * R, 3 * R]
# Pf = 5 * np.eye(3)  # n x n matrix
#
# (risk_type, alpha) = ("AVaR", 0.5)
# problem = r.core.RAOCP(scenario_tree=tree)\
#     .with_markovian_dynamics(As, Bs)\
#     .with_markovian_costs(cost_types, Qs, Rs) \
#     .with_all_leaf_costs(cost_type, Pf)\
#     .with_all_risks(risk_type, alpha)
#
# # print(problem)
#
# solver = r.core.Solver(problem_spec=problem)
# initial_state = np.array([[2], [1], [-1]])
# alpha_cp = .1
# solver.chock(initial_state=initial_state, alpha1=alpha_cp, alpha2=alpha_cp, max_iters=100, tol=1e-5)
