import raocp.core
import numpy as np


def _check_probability_vector(p):
    if abs(sum(p)-1) >= 1e-10:
        raise ValueError("probability vector does not sum up to 1")
    if any(pi <= -1e-16 for pi in p):
        raise ValueError("probability vector contains negative entries")
    return True


class ScenarioTree:

    def __init__(self, stages, ancestors, probability, w_values):
        """
        :param stages:
        :param ancestors:
        :param probability:
        :param w_values: values of w at each node

        Note: avoid using this constructor directly; use a factory instead
        """
        self.__stages = stages
        self.__ancestors = ancestors
        self.__probability = probability
        self.__w_idx = w_values
        self.__children = None  # this will be updated later (the user doesn't need to provide it)
        self.__data = None  # really, any data associated with the nodes of the tree
        self.__update_children()

    def __update_children(self):
        pass

    def num_nodes(self):
        raise NotImplementedError()

    def num_stages(self):
        raise NotImplementedError("#stages not implemented")

    def ancestor_of(self, node_idx):
        return self.__ancestors[node_idx]

    def children_of(self, node_idx):
        raise NotImplementedError()

    def stage_of(self, node_idx):
        raise NotImplementedError()

    def nodes_at_stage(self, stage_idx):
        raise NotImplementedError()

    def probability_of_node(self, node_idx):
        raise NotImplementedError()

    def siblings_of_node(self, node_idx):
        raise NotImplementedError()

    def conditional_probabilities_of_children(self, node_idx):
        raise NotImplementedError()


class MarkovChainScenarioTreeFactory:

    def __init__(self, transition_prob, initial_distribution, num_stages, stopping_time=None):
        self.__transition_prob = transition_prob
        self.__initial_distribution = initial_distribution
        self.__num_stages = num_stages
        self.__stopping_time = stopping_time
        # --- check correctness of `transition_prob` and `initial_distribution`
        for pi in transition_prob:
            _check_probability_vector(pi)
        _check_probability_vector(initial_distribution)

    # function
    # cover = find(chainProbTransMatrix(i,:)~ = 0);
    def __cover(self, i):
        pi = self.__transition_prob[i, :]
        return np.flatnonzero(pi)

    def __make_ancestors_values_stages(self):
        """

        :return: ancestors, values of w and stages
        """
        num_nonzero_init_distr = len(list(filter(lambda x: (x > 0), self.__initial_distribution)))
        # Initialise `ancestors`
        ancestors = np.zeros((num_nonzero_init_distr+1, ))
        ancestors[0] = np.nan  # node 0 does not have an ancestor
        # Initialise `values`
        values = np.zeros((num_nonzero_init_distr+1, ))
        values[0] = np.nan
        values[1:] = np.flatnonzero(self.__initial_distribution)
        # Initialise `stages`
        stages = np.ones((num_nonzero_init_distr+1, ))
        stages[0] = 0

        cursor = 1
        num_nodes_at_stage = num_nonzero_init_distr
        for stage in range(1, self.__stopping_time):
            nodes_added_at_stage = 0
            cursor_new = cursor + num_nodes_at_stage
            for i in range(num_nodes_at_stage):
                node_id = cursor + i
                cover = self.__cover(int(values[node_id]))
                length_cover = len(cover)
                ancestors = np.concatenate((ancestors, node_id * np.ones((length_cover, ))))
                nodes_added_at_stage += length_cover
                values = np.concatenate((values, cover))
            num_nodes_at_stage = nodes_added_at_stage
            cursor = cursor_new
        return ancestors, values, stages

    def __make_probability_values(self):
        return 0

    def create(self):
        # check input data
        ancestors, values, stages = self.__make_ancestors_values_stages()
        probs = self.__make_probability_values()
        tree = ScenarioTree(stages, ancestors, probs, values)
        print(ancestors)
        return tree






