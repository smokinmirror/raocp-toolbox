import raocp as r
import numpy as np


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

# RAOCP generation ----------------------------------------------------------------------------------------------

x0 = np.array([[1], [1]])

Aw1 = np.eye(2)
Aw2 = np.eye(2)
Aw3 = np.eye(2)
As = [Aw1, Aw2, Aw3]

Bw1 = np.eye(2)
Bw2 = np.eye(2)
Bw3 = np.eye(2)
Bs = [Bw1, Bw2, Bw3]

cost_type = "quadratic"
(Q, R, Pf) = (np.eye(2), np.eye(2), np.eye(2))

(risk_type, alpha) = ("AVAR", 0.5)
(E, F, Kone, b) = (np.eye(2), np.eye(2), 0, np.ones((2, 1)))

problem_config = r.core.RAOCPconfig(scenario_tree=tree)\
    .with_possible_As_and_Bs(As, Bs)\
    .with_all_cost_type(cost_type).with_all_Q(Q).with_all_R(R).with_all_Pf(Pf)\
    .with_all_risk_type(risk_type).with_all_alpha(alpha).with_all_E(E).with_all_F(F).with_all_Kone(Kone).with_all_b(b)

problem = r.core.MarkovChainRAOCPFactory(problem_config=problem_config, root_state=x0).create()
