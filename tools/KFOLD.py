import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv


def KFold_Data(urlz, nor=True, n_splits=2):
    """
    将原始数据集进行五折交叉验证的划分
    :param urlz: 需要处理的数据集地址
    :param nor: 是否归一化
    :param n_splits: 划分的折数，默认5折
    :return: 五折交叉验证的数据集，每个元素包含train和test数据
    """
    # 读取数据集
    df = pd.read_csv(urlz, header=None)
    data = df.values
    numberSample, numberAttribute = data.shape
    
    # 数据预处理
    if nor:
        minMax = MinMaxScaler()  # 将数据进行归一化
        U = np.hstack((minMax.fit_transform(data[0:numberSample, 1:]), data[0:numberSample, 0].reshape(numberSample, 1)))
    else:
        U = np.hstack(((data[0:numberSample, 1:]), data[0:numberSample, 0].reshape(numberSample, 1)))
    
    # 标签处理：确保标签只有1和-1
    for i in range(len(U)):
        if U[i][-1] != 1:
            U[i][-1] = -1
    
    # 打乱数据
    np.random.shuffle(U)
    
    # 计算每折的大小
    fold_size = numberSample // n_splits
    folds = []
    
    # 生成五折交叉验证数据集
    for i in range(n_splits):
        # 计算当前折的测试集索引范围
        start = i * fold_size
        end = (i + 1) * fold_size
        
        # 生成测试集
        test = U[start:end]
        
        # 生成训练集：除了当前折的测试集，其余都是训练集
        train = np.vstack((U[:start], U[end:]))
        
        # 添加到结果列表
        folds.append({
            'train': train,
            'test': test,
            'fold_index': i+1
        })
    
    return folds


def Save_KFold_Data(folds, url_prefix, data_name):
    """
    保存五折交叉验证数据集到文件
    :param folds: 五折交叉验证的数据集
    :param url_prefix: 保存路径前缀
    :param data_name: 数据集名称
    :return: None
    """
    for fold in folds:
        fold_index = fold['fold_index']
        train_data = fold['train']
        test_data = fold['test']
        
        # 保存训练集
        train_url = f"{url_prefix}\{data_name}_fold{fold_index}_train.csv"
        with open(train_url, "w", newline='', encoding="utf-8") as jg:
            cw = csv.writer(jg)
            cw.writerows(train_data)
        
        # 保存测试集
        test_url = f"{url_prefix}\{data_name}_fold{fold_index}_test.csv"
        with open(test_url, "w", newline='', encoding="utf-8") as jg:
            cw = csv.writer(jg)
            cw.writerows(test_data)


# 示例用法
"""
urlz = "D:\py\粒球SVM_精度\数据\sonar.csv"
name = "sonar"
nor = True
n_splits = 5

# 生成五折交叉验证数据集
folds = KFold_Data(urlz, nor, n_splits)

# 保存数据集
url_prefix = "D:\py\粒球SVM_精度\划分后数据"
Save_KFold_Data(folds, url_prefix, name)
"""
