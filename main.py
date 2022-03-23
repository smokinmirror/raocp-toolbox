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

root_state = np.array([[1],
                       [1]])  # n vector

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

(risk_type, alpha) = ("AVAR", 0.5)
E = np.eye(2)  # p x n matrix (mu is in R^n)
F = np.eye(2)  # p x r matrix
cone = "Rn+"
b = np.ones((2, 1))  # p vector

problem = r.core.MarkovChainRaocpProblemBuilder(scenario_tree=tree)\
    .with_root_state(root_state)\
    .with_possible_As_and_Bs(As, Bs)\
    .with_all_cost_type(cost_type).with_all_Q(Q).with_all_R(R).with_all_Pf(Pf)\
    .with_all_risk_type(risk_type).with_all_alpha(alpha).with_all_E(E).with_all_F(F).with_all_cone(cone).with_all_b(b)\
    .create()

print(problem)
print(problem.A_at_node(4))
