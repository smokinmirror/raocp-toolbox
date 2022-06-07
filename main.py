import raocp as r
import numpy as np
import raocp.core.nodes as nodes
import raocp.core.dynamics as dynamics
import raocp.core.costs as costs
import raocp.core.risks as risks
import raocp.core.constraints.rectangle as rectangle

# ScenarioTree generation ----------------------------------------------------------------------------------------------

p = np.array([[0.1, 0.8, 0.1],
              [0.4, 0.6, 0],
              [0, 0.3, 0.7]])

v = np.array([0.2, 0.6, 0.2])

(N, tau) = (3, 1)
tree = r.core.MarkovChainScenarioTreeFactory(transition_prob=p,
                                             initial_distribution=v,
                                             num_stages=N, stopping_time=tau).create()

# tree.bulls_eye_plot(dot_size=5, radius=300)
# print(sum(tree.probability_of_node(tree.nodes_at_stage(8))))

# print(tree)

# RAOCP generation -----------------------------------------------------------------------------------------------------
(nl, l) = nodes.Nonleaf(), nodes.Leaf()
(num_states, num_inputs) = 4, 3
factor = 0.1
Aw = factor * np.random.randn(num_states)
Bw = factor * np.random.randn(num_inputs)
for i in range(num_states - 1):
    Aw = np.vstack((Aw, factor * np.random.randn(num_states)))
    Bw = np.vstack((Bw, factor * np.random.randn(num_inputs)))

As = [1.5 * Aw, Aw, -1.5 * Aw]  # n x n matrices
Bs = [-1.5 * Bw, Bw, 1.5 * Bw]  # n x u matrices
mark_dynamics = [dynamics.Dynamics(As[0], Bs[0]),
                 dynamics.Dynamics(As[1], Bs[1]),
                 dynamics.Dynamics(As[2], Bs[2])]

Q = factor * np.eye(num_states)  # n x n matrix
Qs = [1 * Q, 2 * Q, 3 * Q]
R = factor * np.eye(num_inputs)  # u x u matrix OR scalar
Rs = [3 * R, 2 * R, 1 * R]
Pf = factor * 2 * np.eye(num_states)  # n x n matrix
mark_nl_costs = [costs.Quadratic(nl, Qs[0], Rs[0]),
                 costs.Quadratic(nl, Qs[1], Rs[1]),
                 costs.Quadratic(nl, Qs[2], Rs[2])]
leaf_cost = costs.Quadratic(l, Pf)

nonleaf_size = num_states + num_inputs
leaf_size = num_states
limit = 0.01
nl_min = -limit * np.ones((nonleaf_size, 1))
nl_max = limit * np.ones((nonleaf_size, 1))
l_min = -limit * np.ones((leaf_size, 1))
l_max = limit * np.ones((leaf_size, 1))
nl_rect = rectangle.Rectangle(nl, nl_min, nl_max)
l_rect = rectangle.Rectangle(l, l_min, l_max)

alpha = .5
risk = risks.AVaR(alpha)

problem = r.core.RAOCP(scenario_tree=tree) \
    .with_markovian_dynamics(mark_dynamics) \
    .with_markovian_nonleaf_costs(mark_nl_costs) \
    .with_all_leaf_costs(leaf_cost) \
    .with_all_risks(risk) \
    .with_all_nonleaf_constraints(nl_rect) \
    .with_all_leaf_constraints(l_rect)

# print(problem)

solver = r.core.Solver(problem_spec=problem)
initial_state = np.random.randn(num_states).reshape(-1, 1)
solver.chock(initial_state=initial_state, max_iters=3000, tol=1e-3)
