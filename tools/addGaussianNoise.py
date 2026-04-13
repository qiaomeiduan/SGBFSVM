# tools/addGaussianNoise.py
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def add_gaussian_noise(data, noise_ratio):
    """
    给数据的特征列添加高斯噪声（仅处理特征列，标签列保持不变）
    :param data: 输入数据（numpy数组，特征列在前，最后一列是标签）
    :param noise_ratio: 高斯噪声的标准差系数（噪声标准差=特征列标准差×noise_ratio）
    :return: 添加高斯噪声后的数据集（numpy数组，格式与输入一致）
    """
    # 1. 输入校验
    if not isinstance(data, np.ndarray):
        raise ValueError("输入data必须是numpy数组！")
    if data.size == 0:
        return data

    # 分离特征列和标签列（最后一列是标签）
    features = data[:, :-1].copy()  # 特征列（深拷贝避免修改原数据）
    labels = data[:, -1].copy()  # 标签列（保持不变）

    # 2. 逐特征列添加高斯噪声
    for col_idx in range(features.shape[1]):
        feature_col = features[:, col_idx]
        # 计算该特征列的标准差（避免除以0）
        col_std = feature_col.std()
        if col_std == 0:  # 特征列所有值相同，无需添加噪声
            continue

        # 生成高斯噪声：均值=0，标准差=col_std × noise_ratio
        noise = np.random.normal(loc=0.0, scale=col_std * noise_ratio, size=len(feature_col))
        # 添加噪声
        features[:, col_idx] = feature_col + noise

        # 可选：将特征值裁剪到[0,1]（适配归一化后的数据，避免值溢出）
        features[:, col_idx] = np.clip(features[:, col_idx], 0, 1)

    # 3. 合并特征列和标签列
    noisy_data = np.hstack([features, labels.reshape(-1, 1)])

    return noisy_data