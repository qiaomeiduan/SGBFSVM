import numpy as np
import warnings

warnings.filterwarnings("ignore")


def add_feature_noise(data, snr):
    """
    给特征列添加指定SNR的高斯噪声（适配Dive_Data输出格式：最后一列是标签，不处理）
    :param data: 输入数据（numpy数组，Dive_Data处理后，最后一列是标签，前n-1列是特征）
    :param snr: 信噪比（Signal-to-Noise Ratio），如0.1、0.5
    :return: 加特征噪声后的数据集（numpy数组，标签列不变）
    """
    # 分离特征列和标签列（关键：最后一列是标签，前n-1列是特征）
    features = data[:, :-1]  # 所有行，除最后一列的特征列
    labels = data[:, -1:]  # 所有行，最后一列的标签列（保持2D形状，方便拼接）

    # 计算每个特征的信号功率（方差）
    signal_var = np.var(features, axis=0)  # 按列（特征）计算方差
    # 避免方差为0导致除以0（替换0为极小值）
    signal_var[signal_var == 0] = 1e-8

    # 计算噪声方差（噪声功率 = 信号功率 / SNR）
    noise_var = signal_var / snr
    noise_std = np.sqrt(noise_var)  # 噪声标准差

    # 生成零均值高斯噪声（形状与特征列完全一致）
    noise = np.random.normal(loc=0, scale=noise_std, size=features.shape)

    # 特征列叠加噪声（仅修改特征，标签不变）
    features_noisy = features + noise

    # 拼接特征列和标签列，返回加噪后的数据（保持格式：特征在前，标签在最后一列）
    data_noisy = np.hstack((features_noisy, labels))
    return data_noisy