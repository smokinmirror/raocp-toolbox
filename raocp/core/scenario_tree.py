class ScenarioTree:

    def __init__(self):
        self.__stage = None
        self.__children = None
        self.__ancestors = None
        self.__probability = None

    def num_nodes(self):
        raise NotImplemented()

    def ancestor_of(self, node_idx):
        raise NotImplemented()

    def children_of(self, node_idx):
        raise NotImplemented()

    def nodes_at_stage(self, stage_idx):
        raise NotImplemented()

    def probability_of_node(self, node_idx):
        raise NotImplemented()

    def siblings_of_node(self, node_idx):
        raise NotImplemented()

    def conditional_probabilities_of_children(self, node_idx):
        raise NotImplemented()

    def get_x(self):
        return self.__x

    @staticmethod
    def from_markov_chain(transition_prob,
                          initial_distribution,
                          stages,
                          stopping_time=None):
        return ScenarioTree()






