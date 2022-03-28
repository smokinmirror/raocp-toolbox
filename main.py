import raocp as r
import numpy as np

# ScenarioTree generation ----------------------------------------------------------------------------------------------

p = np.array([[0.1, 0.8, 0.1],
              [0.4, 0.6, 0],
              [0, 0.3, 0.7]])

v = np.array([0.5, 0.4, 0.1])

(N, tau) = (8, 5)
tree = r.core.MarkovChainScenarioTreeFactory(transition_prob=p,
                                             initial_distribution=v,
                                             num_stages=N, stopping_time=tau).create()

# tree.bulls_eye_plot(dot_size=5, radius=300)
#
# tree.set_data_at_node(4, {"a": 1})
# print(sum(tree.probability_of_node(tree.nodes_at_stage(8))))
#
# print(tree)

# RAOCP generation -----------------------------------------------------------------------------------------------------

Aw1 = np.eye(2)
Aw2 = 2*np.eye(2)
Aw3 = 3*np.eye(2)
As = [Aw1, Aw2, Aw3]  # n x n matrices

Bw1 = np.eye(2)
Bw2 = 2*np.eye(2)
Bw3 = 3*np.eye(2)
Bs = [Bw1, Bw2, Bw3]  # n x u matrices

cost_type = "quadratic"
Q = 10*np.eye(2)  # n x n matrix
R = np.eye(2)  # u x u matrix OR scalar
Pf = 5*np.eye(2)  # n x n matrix

(risk_type, alpha) = ("AVaR", 0.5)
problem = r.core.MarkovChainRAOCPProblemBuilder(scenario_tree=tree)\
    .with_possible_As_and_Bs(As, Bs)\
    .with_all_cost(cost_type, Q, R, Pf)\
    .with_all_risk(risk_type, alpha)\
    .create()

print(problem)
