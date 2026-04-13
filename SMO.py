"""
    Author: Lasse Regin Nielsen
"""

from __future__ import division, print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

from numpy import sqrt


filepath = os.path.dirname(os.path.abspath(__file__))


class SMO():
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """

    def __init__(self, max_iter=3000, kernel_type='quadratic', C=1.0, me=0, epsilon=0.00001):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic,
            'gaussian': self.kernel_gaussian
        }
        self.me = me
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C * me,
        self.epsilon = epsilon

    def fit(self, X, R, y):
        # 初始化参数
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros(n)

        # 1. 预计算核矩阵 (Precompute Kernel Matrix)
        # 避免在循环中重复调用函数计算 kernel(x_i, x_j)
        if self.kernel_type == 'linear':
            K = np.dot(X, X.T)
        elif self.kernel_type == 'quadratic':
            K = (np.dot(X, X.T) + 1) ** 2
        elif self.kernel_type == 'gaussian':
            # 高斯核计算稍微复杂，建议对于小样本(如256)预计算，大样本则动态计算
            sq_norms = np.sum(X ** 2, axis=1)
            dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(X, X.T)
            gamma = 1.0  # 默认值
            K = np.exp(-gamma * dist_sq)
        else:
            # 回退到原始慢速计算（如果是不常用的自定义核）
            kernel = self.kernels[self.kernel_type]
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = kernel(X[i], X[j])

        # 2. 初始化全局变量 A_vec 和 B_val (增量维护的对象)
        # A_vec = sum(alpha_k * y_k * X_k)
        # B_val = sum(alpha_k * R_k)
        A_vec = np.zeros(d)
        B_val = 0.0

        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)

            for j in range(0, n):
                i = self.get_rnd_int(0, n - 1, j)

                # 提取基础参数
                x_i, x_j, y_i, y_j = X[i, :], X[j, :], y[i], y[j]
                k11, k12, k22 = K[i, i], K[i, j], K[j, j]
                k_ij = k11 + k22 - 2 * k12

                if k_ij == 0:
                    continue

                # --- 向量化计算 sum_K 和 sum_R ---
                # 原本的 for ii 循环
                # 计算时排除当前处理的 i 和 j
                mask = np.ones(n, dtype=bool)
                mask[[i, j]] = False

                # sum_K1 = sum(alpha_ii * y_ii * K_i_ii)
                sum_K1 = np.dot(alpha[mask] * y[mask], K[i, mask])
                sum_K2 = np.dot(alpha[mask] * y[mask], K[j, mask])

                # sum_R1 = R[i] * sum(R_ii), sum_R2 = R[j] * sum(R_ii)
                R_sum_others = np.sum(R[mask])
                sum_R1 = R[i] * R_sum_others
                sum_R2 = R[j] * R_sum_others

                # --- 增量计算 lambda1 ---
                # 使用向量模长计算 A
                A_norm = np.linalg.norm(A_vec)
                lambda1 = (2 * A_norm) / (B_val + 1e-10) - 1

                # --- 更新 Alpha ---
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                C_i, C_j = self.C[0][i], self.C[0][j]

                (L, H) = self.compute_L_H(C_i, C_j, alpha_prime_j, alpha_prime_i, y_j, y_i)

                R11, R12, R22 = R[i] * R[i], R[i] * R[j], R[j] * R[j]
                s = y_i * y_j

                # 计算 eta 和 a2_1
                eta = k11 + k22 - 2 * k12 - lambda1 * R11 - lambda1 * R22 + 2 * y_i * y_j * R12
                kesi = alpha_prime_i * y_i + alpha_prime_j * y_j

                a2_1 = (kesi * y_j * k11 - kesi * y_j * k12 + y_j * sum_K1 - y_j * sum_K2 -
                        lambda1 * kesi * y_j * R11 + lambda1 * kesi * y_i * R12 -
                        lambda1 * s * sum_R1 + lambda1 * sum_R2 - s + 1)

                # 更新 alpha[j] 并剪裁
                alpha[j] = a2_1 / eta
                alpha[j] = np.clip(alpha[j], L, H)

                # 更新 alpha[i]
                alpha[i] = alpha_prime_i + y_i * y_j * (alpha_prime_j - alpha[j])

                # --- 关键优化：增量更新 A_vec 和 B_val ---
                # 每一轮 j 循环结束后，根据 alpha 的变化量同步更新 A 和 B，无需重新全量求和
                diff_i = (alpha[i] - alpha_prime_i) * y_i
                diff_j = (alpha[j] - alpha_prime_j) * y_j

                A_vec += diff_i * x_i + diff_j * x_j
                B_val += (alpha[i] - alpha_prime_i) * R[i] + (alpha[j] - alpha_prime_j) * R[j]

            # 检查收敛性
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                print(f"Convergence reached at iteration {count}")
                break

            if count >= self.max_iter:
                print(f"Max iterations {self.max_iter} reached")
                break

        # 计算最终模型参数
        self.w = self.calc_w(alpha, y, X, R)
        self.b = self.calc_b(X, y, self.w)

        alpha_idx = np.where(alpha > 0)[0]
        return X[alpha_idx, :], count

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calc_b(self, X, y, w, torch=None):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X, R):
        # print('alpha***********',alpha)
        A_2 = np.zeros(len(X[0]))
        B_2 = 0
        t_x2 = 0
        for k in range(len(alpha)):
            A_2 = A_2 + alpha[k] * y[k] * X[k]
            B_2 = B_2 + alpha[k] * R[k]
            t_x2 += alpha[k]
        # A_1 = np.sqrt(np.sum([i * i for i in A]))
        A_22 = np.sqrt(np.sum(A_2 ** 2))
        w = ((A_22 - B_2) * A_2) / A_22
        return w

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C_i, C_j, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if (y_i != y_j):
            # (max(0, alpha_prime_j - alpha_prime_i), min(C,  C - alpha_prime_i + alpha_prime_j ))
            return (max(0, alpha_prime_j - alpha_prime_i), min(C_j, C_i - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C_i), min(C_j, alpha_prime_i + alpha_prime_j))

    #(max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_rnd_int(self, a, b, z):
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt = cnt + 1
        return i

    # Define kernels定义核函数
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)

    def kernel_gaussian(self, x, y, sigma=1):
        # 高斯核函数
        """Returns the gaussian similarity of arrays 'x' and 'y' with
        kernel width paramenter 'sigma' (set to 1 by default)"""

        if np.ndim(x) == 1 and np.ndim(y) == 1:
            result = np.exp(-(np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
        elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
            result = np.exp(-(np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
        elif np.ndim(x) > 1 and np.ndim(y) > 1:
            result = np.exp(-(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
        return result
