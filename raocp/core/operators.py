import numpy as np
from scipy.linalg import sqrtm
import raocp.core.problem_spec as ps


class Operator:
    """
    Linear operators
    """

    def __init__(self, problem_spec: ps.RAOCP, initial_primal, primal_split, initial_dual, dual_split):
        self.__raocp = problem_spec
        self.__num_nonleaf_nodes = self.__raocp.tree.num_nonleaf_nodes
        self.__num_nodes = self.__raocp.tree.num_nodes
        self.__primal = initial_primal
        self.__primal_split = primal_split
        self.__dual = initial_dual
        self.__dual_split = dual_split

        # create initial sections for writing into

        # primal
        self.__states = self.__primal[self.__primal_split[1]: self.__primal_split[2]]  # x
        self.__controls = self.__primal[self.__primal_split[2]: self.__primal_split[3]]  # u
        self.__dual_risk_y = self.__primal[self.__primal_split[3]: self.__primal_split[4]]  # y
        self.__relaxation_tau = self.__primal[self.__primal_split[4]: self.__primal_split[5]]  # tau
        self.__relaxation_s = self.__primal[self.__primal_split[5]: self.__primal_split[6]]  # s
        # dual
        self.__dual_1 = self.__dual[self.__dual_split[1]: self.__dual_split[2]]
        self.__dual_2 = self.__dual[self.__dual_split[2]: self.__dual_split[3]]
        self.__dual_3 = self.__dual[self.__dual_split[3]: self.__dual_split[4]]
        self.__dual_4 = self.__dual[self.__dual_split[4]: self.__dual_split[5]]
        self.__dual_5 = self.__dual[self.__dual_split[5]: self.__dual_split[6]]
        self.__dual_6 = self.__dual[self.__dual_split[6]: self.__dual_split[7]]
        self.__dual_7 = self.__dual[self.__dual_split[7]: self.__dual_split[8]]
        self.__dual_11 = self.__dual[self.__dual_split[11]: self.__dual_split[12]]
        self.__dual_12 = self.__dual[self.__dual_split[12]: self.__dual_split[13]]
        self.__dual_13 = self.__dual[self.__dual_split[13]: self.__dual_split[14]]
        self.__dual_14 = self.__dual[self.__dual_split[14]: self.__dual_split[15]]

    def ell(self, modified_primal):
        # create sections for ease of access
        self.__states = modified_primal[self.__primal_split[1]: self.__primal_split[2]]
        self.__controls = modified_primal[self.__primal_split[2]: self.__primal_split[3]]
        self.__dual_risk_y = modified_primal[self.__primal_split[3]: self.__primal_split[4]]
        self.__relaxation_tau = modified_primal[self.__primal_split[4]: self.__primal_split[5]]
        self.__relaxation_s = modified_primal[self.__primal_split[5]: self.__primal_split[6]]

        for i in range(self.__num_nonleaf_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            self.__dual_1[i] = self.__dual_risk_y[i]
            self.__dual_2[i] = self.__relaxation_s[stage_at_i][i] \
                - self.__raocp.risk_at_node(i).vector_b.T @ self.__dual_risk_y[i]
            for j in children_of_i:
                self.__dual_3[j] = sqrtm(self.__raocp.nonleaf_cost_at_node(j).nonleaf_state_weights) @ self.__states[i]
                self.__dual_4[j] = sqrtm(self.__raocp.nonleaf_cost_at_node(j).control_weights) @ self.__controls[i]
                half_tau = 0.5 * self.__relaxation_tau[stage_at_children_of_i][j]
                self.__dual_5[j] = half_tau
                self.__dual_6[j] = half_tau

            self.__dual_7[i] = self.__raocp.

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            self.__dual_11[i] = sqrtm(self.__raocp.leaf_cost_at_node(i).leaf_state_weights) @ self.__states[i]
            half_s = 0.5 * self.__relaxation_s[stage_at_i][i]
            self.__dual_12[i] = half_s
            self.__dual_13[i] = half_s

        # collect modified sections of dual
        self.__dual = self.__dual_1 + self.__dual_2 + self.__dual_3 + self.__dual_4 + self.__dual_5 + \
            self.__dual_6 + self.__dual_7 + self.__dual_11 + self.__dual_12 + self.__dual_13 + self.__dual_14

        return self.__dual

    def ell_transpose(self, modified_dual):
        # create sections for ease of access
        self.__dual_1 = modified_dual[self.__dual_split[1]: self.__dual_split[2]]
        self.__dual_2 = modified_dual[self.__dual_split[2]: self.__dual_split[3]]
        self.__dual_3 = modified_dual[self.__dual_split[3]: self.__dual_split[4]]
        self.__dual_4 = modified_dual[self.__dual_split[4]: self.__dual_split[5]]
        self.__dual_5 = modified_dual[self.__dual_split[5]: self.__dual_split[6]]
        self.__dual_6 = modified_dual[self.__dual_split[6]: self.__dual_split[7]]
        self.__dual_11 = modified_dual[self.__dual_split[7]: self.__dual_split[8]]
        self.__dual_12 = modified_dual[self.__dual_split[8]: self.__dual_split[9]]
        self.__dual_13 = modified_dual[self.__dual_split[9]: self.__dual_split[10]]

        for i in range(self.__num_nonleaf_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            self.__dual_risk_y[i] = (self.__dual_1[i] - self.__raocp.risk_at_node(i).vector_b @ self.__dual_2[i]) \
                .reshape((2 * self.__raocp.tree.children_of(i).size + 1, 1))  # reshape to column vector
            self.__relaxation_s[stage_at_i][i] = self.__dual_2[i]
            self.__states[i] = 0
            self.__controls[i] = 0
            for j in children_of_i:
                self.__states[i] += sqrtm(self.__raocp.nonleaf_cost_at_node(j).nonleaf_state_weights).T \
                                    @ self.__dual_3[j]
                self.__controls[i] += sqrtm(self.__raocp.nonleaf_cost_at_node(j).control_weights).T @ self.__dual_4[j]
                self.__relaxation_tau[stage_at_children_of_i][j] = 0.5 * (self.__dual_5[j] + self.__dual_6[j])

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            self.__states[i] = sqrtm(self.__raocp.leaf_cost_at_node(i).leaf_state_weights).T @ self.__dual_11[i]
            self.__relaxation_s[stage_at_i][i] = 0.5 * (self.__dual_12[i] + self.__dual_13[i])

        # collect modified sections of dual
        self.__primal = self.__states + self.__controls + self.__dual_risk_y + \
            self.__relaxation_tau + self.__relaxation_s

        return self.__primal
