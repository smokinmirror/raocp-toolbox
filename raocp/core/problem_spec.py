from typing import Union, Any

import numpy as np
import turtle


class RAOCP:
    """
    Risk-averse optimal control problem creation
    """

    def __init__(self, state, ip, system_dynamics, input_dynamics,
                 cost_type, Q, R, Pf,
                 risk_type, alpha):
        """
        :param state: list of the state (x) at each node
        :param ip: list of the input (u) at each node
        :param system_dynamics: list of the system dynamics (A) at each node
        :param input_dynamics: list of the input dynamics (B) at each node
        :param cost_type: list of the cost type (quad, ...) at each node
        :param Q: list of the Q matrix used for computing nonleaf node cost
        :param R: list of the R matrix used for computing nonleaf node cost
        :param Pf: list of the P_f matrix used for computing leaf node cost
        :param risk_type: list of the risk type (AVAR, EVAR, ...) at each node
        :param alpha: list of risk parameter at each node

        Note: avoid using this constructor directly; use a factory instead
        """
        # System
        self.__state = state
        self.__input = ip
        self.__system_dynamics = system_dynamics
        self.__input_dynamics = input_dynamics
        # Cost
        self.__cost_type = cost_type
        self.__Q = Q
        self.__R = R
        self.__Pf = Pf
        self.__cost = None
        self.__update_cost()
        # Risk
        self.__risk_type = risk_type
        self.__alpha = alpha
        self.__risk = None
        self.__update_risk()

    def __update_cost(self):  # currently only supporting quadratic cost
        raise NotImplementedError()

    def __update_risk(self):
        raise NotImplementedError()

    def __terminal_cost(self):
        raise NotImplementedError()

    def __dynamics(self):
        raise NotImplementedError()

    def __constraints(self):
        raise NotImplementedError()

    def __risks(self):
        raise NotImplementedError()


class MarkovChainRAOCPFactory:
    """
    Factory class to construct risk-averse optimal control problem from stopped Markov chains
    """

    def __init__(self, x_current, u_current, data, w_value=None):
        """
        :param x_current: state in node
        :param u_current: input in node
        :param data: data in node
        :param w_value: w_value in node [default: None]
        """
        self.__x_current = x_current
        self.__u_current = u_current
        self.__data = data
        self.__w_value = w_value
        # self.__cost_type = data["cost"]["type"]
        # self.__cost_Q = data["cost"]["Q"]
        # self.__cost_R = data["cost"]["R"]
        # self.__constraints_type = data["constraints"]["type"]
        # self.__constraints_x_min = data["constraints"]["x_min"]
        # self.__constraints_x_min = data["constraints"]["x_max"]
        # self.__constraints_x_min = data["constraints"]["u_min"]
        # self.__constraints_x_min = data["constraints"]["u_max"]
        # self.__dynamics_type = data["dynamics"]["type"]
        # self.__dynamics_A = data["dynamics"]["A"]
        # self.__dynamics_B = data["dynamics"]["B"]
        # self.__risk_type = data["risk"]["type"]
        # self.__risk_alpha = data["risk"]["alpha"]
        # self.__risk_E = data["risk"]["E"]
        # self.__risk_F = data["risk"]["F"]
        # self.__risk_b = data["risk"]["b"]

    def __make_cost(self):
        """
        :return: cost
        """
        if self.__data["cost"]["type"] == "quadratic":
            cost = self.__x_current.T @ self.__data["cost"]["Q"] @ self.__x_current \
                   + self.__u_current.T @ self.__data["cost"]["R"] @ self.__u_current
        else:
            raise ValueError("cost type is not quadratic!")
        return cost

    def __make_linear_func(self):
        """
        :return: linear result
        """
        if self.__data["dynamics"]["type"] == "affine":
            x_next = self.__data["dynamics"]["A"] @ self.__x_current + \
                     self.__data["dynamics"]["B"] @ self.__u_current + self.__w_value
        elif self.__data["dynamics"]["type"] == "linear":
            x_next = self.__data["dynamics"]["A"] @ self.__x_current + self.__w_value
        else:
            raise ValueError("dynamics type is not affine or linear!")
        return x_next

    def __make_risk_value(self):
        """
        :return: risk results
        """
        if self.__data["risk"]["type"] == "AV@R":
            raise NotImplementedError()
        else:
            raise ValueError("risk type is not AV@R!")

    def create(self):
        """
        Creates a Risk-averse optimal control problem from the given Markov chain
        """
        x_next = self.__make_linear_func()
        cost = self.__make_cost()
        risk_value = self.__make_risk_value()
        raocp = RaOCP(x_next, cost, risk_value)
        return raocp
