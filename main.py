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

tree.bulls_eye_plot(dot_size=5, radius=300)

tree.set_data_at_node(4, {"a": 1})
print(sum(tree.probability_of_node(tree.nodes_at_stage(8))))

print(tree)
