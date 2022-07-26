# import numpy as np
# # from scipy.sparse.linalg import LinearOperator, eigs
# import raocp.core.cache as cache
# import raocp.core.solver as solver
#
#
# class SuperMann:
#     """
#     SuperMann accelerator
#     """
#
#     def __init__(self, cache_: cache.Cache, c0, c1, c2, beta, sigma, lamda):
#         self.__solver = None
#         self.__cache = cache_
#         self.__c0 = c0
#         self.__c1 = c1
#         self.__c2 = c2
#         self.__beta = beta
#         self.__sigma = sigma
#         self.__lamda = lamda
#         self.__eta = [None] * 2
#         self.__w = [None] * 2
#         self.__pos_k = 0
#         self.__pos_kplus1 = 1
#         self.__r_safe = None
#         self.__counter_k = 0
#
#     def run(self, solver_: solver.Solver):
#         self.__solver = solver_
#         self._chock_operator()
#         current_error = solver.get_current_error()
#         return current_error
#
#     def _chock_operator(self):
#         # run primal part of algorithm
#         self.__solver.primal_k_plus_half()
#         self.__solver.primal_k_plus_one()
#         # run dual part of algorithm
#         self.__solver.dual_k_plus_half()
#         self.__solver.dual_k_plus_one()
#
#     @staticmethod
#     def _parts_to_vector(prim_, dual_):
#         return np.vstack((np.vstack(prim_), np.vstack(dual_)))
#
#     def _vector_to_parts(self, vector_):
#         prim_ = self.__cache.get_primal()
#         dual_ = self.__cache.get_dual()
#         index = 0
#         for i in range(len(prim_)):
#             size_ = prim_[i].size
#             prim_[i] = vector_[index: size_]
#             index += size_
#
#         for i in range(len(dual_)):
#             size_ = dual_[i].size
#             dual_[i] = vector_[index: size_]
#             index += size_
#
#         return prim_, dual_
