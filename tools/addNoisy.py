import pandas as pd
import collections
import copy
import warnings
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def recreat_data(train_data, noise_ratio):
    """
    给训练集添加标签噪声（仅适配五折交叉验证的训练集：四折拼接的数据），测试集请勿调用此函数
    :param train_data: 训练集数据（numpy数组，格式：特征列在前，最后一列是标签，标签值为1/-1）
    :param noise_ratio: 标签噪声比例（如0.1表示10%的样本置换标签）
    :return: 加标签噪声后的训练集（numpy数组，格式与输入一致）
    """
    # 1. 输入校验（确保是numpy数组，且非空）
    if not isinstance(train_data, np.ndarray):
        raise ValueError("输入train_data必须是numpy数组！")
    if train_data.size == 0:
        return train_data

    numSamples, numAttribute = train_data.shape
    # 2. 计算需要置换标签的样本总数（避免小数，向下取整）
    total_noise_samples = int(numSamples * noise_ratio)
    if total_noise_samples == 0:  # 噪声比例为0时，直接返回原数据
        return train_data

    # 3. 转换为DataFrame方便按标签筛选
    df = pd.DataFrame(train_data)
    label_col = numAttribute - 1  # 标签列是最后一列

    # 4. 统计标签分布（兼容1/-1二分类）
    label_counter = collections.Counter(train_data[:, label_col])
    label_types = list(label_counter.keys())  # 标签类别（如[1, -1]）
    label_counts = list(label_counter.values())  # 各类别样本数

    # 5. 按标签类别分配噪声样本数（按原分布比例）
    noise_dict = {}  # 存储每个标签类别需要置换的样本数
    for i, label in enumerate(label_types):
        noise_dict[label] = int(total_noise_samples * (label_counts[i] / numSamples))

    # 6. 按类别处理标签置换
    df_dict = {}
    for label in label_types:
        # 筛选当前标签的样本，重置索引方便修改
        df_label = df[df[label_col] == label].reset_index(drop=True)
        df_dict[label] = df_label

        # 计算当前类别需要置换的样本数
        need_noise = noise_dict[label]
        if need_noise == 0 or len(label_types) < 2:  # 无噪声需求/单标签，跳过
            continue

        # 目标标签：排除当前标签，随机选（二分类时直接选另一个）
        target_labels = [l for l in label_types if l != label]
        # 按目标标签的样本数比例分配置换数量
        target_counts = [label_counter[l] for l in target_labels]
        target_total = sum(target_counts)

        if target_total == 0:  # 无其他标签样本，跳过
            continue

        # 遍历目标标签，置换样本
        start_idx = 0
        for t_label, t_count in zip(target_labels, target_counts):
            # 计算当前目标标签需要置换的样本数
            t_noise = int(need_noise * (t_count / target_total))
            if t_noise == 0:
                continue
            # 置换标签（仅修改指定行的标签列）
            end_idx = start_idx + t_noise
            if end_idx > len(df_label):  # 避免索引越界
                end_idx = len(df_label)
            df_label.loc[start_idx:end_idx - 1, label_col] = t_label
            start_idx = end_idx

        # 更新置换后的标签数据
        df_dict[label] = df_label

    # 7. 合并所有标签类别的数据
    new_df = pd.concat(df_dict.values(), ignore_index=True)
    # 8. 随机打乱（保持样本顺序随机性）
    new_df = shuffle(new_df).reset_index(drop=True)

    # 9. 转换回numpy数组，保持格式一致
    return new_df.values