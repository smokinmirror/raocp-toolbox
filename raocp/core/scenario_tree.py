<<<<<<< HEAD
import raocp.core
import numpy as np


class ScenarioTree:

    def __init__(self):
        self.__stage = None
        self.__children = None
        self.__ancestors = None
        self.__probability = None

    def num_nodes(self):
        return len(self.__ancestors)

    def ancestor_of(self, node_idx):
        return self.__ancestors[node_idx]

    def children_of(self, node_idx):
        raise NotImplemented()

    def nodes_at_stage(self, stage_idx):
        return self.__stage[stage_idx]

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
        tree = ScenarioTree()
        # tree.__ancestors = [None, 0, 0, 1, 1, 1, 123, 456]

        # ------ construct first stage
        tree.__ancestors = [None]
        tree.__stage = [0]
        node_index = [0]
        init_state_num = len(initial_distribution)
        n_init_dist_non_zero = np.count_nonzero(initial_distribution)  # count nonzero of initial_distribution
        n_init_dist_non_zero_array = [1, 1, 0]  # todo: get the numbers of nonzero of initial_distribution
        # branch_num = np.count_nonzero(transition_prob)  # nonzero of transition_prob
        branch_num = [3, 2, 2]  # todo: get the numbers of nonzero of transition_prob
        nodes_at_stage = n_init_dist_non_zero
        current_ancestors = -1
        cursor_of_node = 1
        children_nodes = [1, 2]  # start from node 0
        # ------ construct the ancestor array for subsequent stages
        for stage in range(stopping_time):
            nodes_added_at_this_stage = 0
            new_cursor_position = cursor_of_node + nodes_at_stage
            nodes_id_at_stage = []  # empty the list every stage
            for i_node in range(nodes_at_stage):
                node_id = cursor_of_node + i_node
                node_index.append(node_id)
                length_node_index = len(node_index)
                nodes_added_at_this_stage = nodes_added_at_this_stage + branch_num[i_node % init_state_num]
                if (i_node % init_state_num) == 0:  # todo: not correct! Should consider branch_num [3, 2, 2]
                    current_ancestors = current_ancestors + 1
                tree.__ancestors.append(current_ancestors)
                nodes_id_at_stage.append(node_id)  # save all the nodes id in stage, use for tree.__stage
            tree.__stage.append(nodes_id_at_stage)
            print(node_index)
            print(length_node_index)
            cursor_of_node = new_cursor_position
            if stage < stopping_time - 1:
                nodes_at_stage = nodes_added_at_this_stage

        # ------ construct the ancestor array for the nonbranching part
        for stage in range(stages - stopping_time):
            # nodes_added_at_this_stage = 0
            # new_cursor_position = cursor_of_node + nodes_at_stage
            nodes_id_at_stage = []  # empty the list every stage
            for i_node in range(nodes_at_stage):
                node_id = cursor_of_node + i_node
                node_index.append(node_id)
                length_node_index = len(node_index)
                nodes_added_at_this_stage = nodes_added_at_this_stage + branch_num[i_node % init_state_num]
                current_ancestors = current_ancestors + 1
                tree.__ancestors.append(current_ancestors)
                nodes_id_at_stage.append(node_id)  # save all the nodes id in stage, use for tree.__stage
            cursor_of_node = cursor_of_node + nodes_at_stage
            tree.__stage.append(nodes_id_at_stage)
        print(node_index)
        print(length_node_index)
        print(tree.__stage)
        print(tree.__ancestors)

        return tree
=======
class ScenarioTree:

    def __init__(self):
        print("Constructed scenario tree")
        pass
>>>>>>> main
