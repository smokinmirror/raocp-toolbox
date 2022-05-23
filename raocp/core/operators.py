import numpy as np
from scipy.linalg import sqrtm
import raocp.core.cache as core_cache


class Operator:
    """
    Linear operators
    """

    def __init__(self, cache: core_cache.Cache):
        self.__raocp = cache.get_raocp()
        self.__num_nonleaf_nodes = self.__raocp.tree.num_nonleaf_nodes
        self.__num_nodes = self.__raocp.tree.num_nodes
        self.__segment_p = cache.get_primal_segments()
        self.__segment_d = cache.get_dual_segments()

    def ell(self, input_primal, output_dual):
        # create sections for ease of access
        states = input_primal[self.__segment_p[1]: self.__segment_p[2]]
        controls = input_primal[self.__segment_p[2]: self.__segment_p[3]]
        dual_risk_y = input_primal[self.__segment_p[3]: self.__segment_p[4]]
        relaxation_tau = input_primal[self.__segment_p[4]: self.__segment_p[5]]
        relaxation_s = input_primal[self.__segment_p[5]: self.__segment_p[6]]

        for i in range(self.__num_nonleaf_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            output_dual[self.__segment_d[1] + i] = dual_risk_y[i]
            output_dual[self.__segment_d[2] + i] = relaxation_s[stage_at_i][i] \
                                                   - self.__raocp.risk_at_node(i).vector_b.T @ dual_risk_y[i]
            for j in children_of_i:
                output_dual[self.__segment_d[3] + j] = \
                    sqrtm(self.__raocp.nonleaf_cost_at_node(j).state_weights) @ states[i]
                output_dual[self.__segment_d[4] + j] = \
                    sqrtm(self.__raocp.nonleaf_cost_at_node(j).control_weights) @ controls[i]
                half_tau = 0.5 * relaxation_tau[stage_at_children_of_i][j]
                output_dual[self.__segment_d[5] + j] = half_tau
                output_dual[self.__segment_d[6] + j] = half_tau

            output_dual[self.__segment_d[7] + i] = self.__raocp.nonleaf_constraint_at_node(i).state_matrix @ states[i] \
                + self.__raocp.nonleaf_constraint_at_node(i).control_matrix @ controls[i]

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            output_dual[self.__segment_d[11] + i] = \
                sqrtm(self.__raocp.leaf_cost_at_node(i).state_weights) @ states[i]
            half_s = 0.5 * relaxation_s[stage_at_i][i]
            output_dual[self.__segment_d[12] + i] = half_s
            output_dual[self.__segment_d[13] + i] = half_s
            output_dual[self.__segment_d[14] + i] = self.__raocp.leaf_constraint_at_node(i).state_matrix @ states[i]

    def ell_transpose(self, input_dual, output_primal):
        # create sections for ease of access
        dual_1 = input_dual[self.__segment_d[1]: self.__segment_d[2]]
        dual_2 = input_dual[self.__segment_d[2]: self.__segment_d[3]]
        dual_3 = input_dual[self.__segment_d[3]: self.__segment_d[4]]
        dual_4 = input_dual[self.__segment_d[4]: self.__segment_d[5]]
        dual_5 = input_dual[self.__segment_d[5]: self.__segment_d[6]]
        dual_6 = input_dual[self.__segment_d[6]: self.__segment_d[7]]
        dual_7 = input_dual[self.__segment_d[7]: self.__segment_d[8]]
        dual_11 = input_dual[self.__segment_d[11]: self.__segment_d[12]]
        dual_12 = input_dual[self.__segment_d[12]: self.__segment_d[13]]
        dual_13 = input_dual[self.__segment_d[12]: self.__segment_d[14]]
        dual_14 = input_dual[self.__segment_d[14]: self.__segment_d[15]]

        for i in range(self.__num_nonleaf_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            output_primal[self.__segment_p[3] + i] = (dual_1[i] - self.__raocp.risk_at_node(i).vector_b @ dual_2[i]) \
                .reshape((2 * self.__raocp.tree.children_of(i).size + 1, 1))  # reshape to column vector
            output_primal[self.__segment_p[5] + stage_at_i][i] = dual_2[i]
            output_primal[self.__segment_p[1] + i] = \
                self.__raocp.nonleaf_constraint_at_node(i).state_matrix.T @ dual_7[i]
            output_primal[self.__segment_p[2] + i] = \
                self.__raocp.nonleaf_constraint_at_node(i).control_matrix.T @ dual_7[i]
            for j in children_of_i:
                output_primal[self.__segment_p[1] + i] = output_primal[self.__segment_p[1] + i] + \
                    (sqrtm(self.__raocp.nonleaf_cost_at_node(j).state_weights) @ dual_3[j])
                output_primal[self.__segment_p[2] + i] = output_primal[self.__segment_p[2] + i] + \
                    (sqrtm(self.__raocp.nonleaf_cost_at_node(j).control_weights) @ dual_4[j])
                output_primal[self.__segment_p[4] + stage_at_children_of_i][j] = 0.5 * (dual_5[j] + dual_6[j])

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            output_primal[self.__segment_p[1] + i] = \
                self.__raocp.leaf_constraint_at_node(i).state_matrix.T @ dual_14[i] + \
                sqrtm(self.__raocp.leaf_cost_at_node(i).state_weights) @ dual_11[i]
            output_primal[self.__segment_p[5] + stage_at_i][i] = 0.5 * (dual_12[i] + dual_13[i])
