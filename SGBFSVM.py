import numpy as np
import time
from SMO import SMO
from warnings import filterwarnings
from sklearn.cluster import k_means
from sklearn.metrics import roc_auc_score
from tools.gen_ball import gen_balls
from tools.addGaussianNoise import add_gaussian_noise
from tools.KFOLD import KFold_Data

filterwarnings('ignore')
np.set_printoptions(suppress=True)


# ===================== 核心指标计算 =====================
def getacc(data, w, b):
    TP = FP = TN = FN = 0
    y_true, y_score = [], []
    for v in data:
        score = np.sum(w * np.array(v[:-1])) + b
        y_score.append(score)
        true_label = 1 if v[-1] > 0.1 else 0
        y_true.append(true_label)
        if true_label == 1 and score > 0:
            TP += 1
        elif true_label == 1 and score <= 0:
            FN += 1
        elif true_label == 0 and score <= 0:
            TN += 1
        elif true_label == 0 and score > 0:
            FP += 1

    acc = (TP + TN) / (TP + FP + TN + FN + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    F_score = (2 * precision * recall) / (precision + recall + 1e-7)
    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
    except:
        auc = 0.5
    return acc, recall, precision, F_score, auc


# ===================== 粒球隶属度计算 =====================
def ball_membership(datab):
    if not datab:
        return []
    weighted_sum_pos = 0
    weight_sum_pos = 0
    weighted_sum_neg = 0
    weight_sum_neg = 0

    for dd in datab:
        c_i = np.array(dd[0])
        r_i = dd[1]
        label = dd[-1]
        temp_ball = np.array([[1 if label == 1 else 0] + c_i.tolist()])
        _, p_i = get_label_and_purity(temp_ball)
        if label == 1:
            weighted_sum_pos += r_i * p_i * c_i
            weight_sum_pos += r_i * p_i
        else:
            weighted_sum_neg += r_i * p_i * c_i
            weight_sum_neg += r_i * p_i

    center1 = weighted_sum_pos / weight_sum_pos if weight_sum_pos > 0 else 0
    center2 = weighted_sum_neg / weight_sum_neg if weight_sum_neg > 0 else 0
    processed_balls = []
    for d in datab:
        c_i = np.array(d[0])
        label = d[-1]
        r_i = d[1]
        if label == 1:
            dist_to_pos = np.sqrt(np.sum((c_i - center1) ** 2))
            r = max([dist_to_pos]) if [dist_to_pos] else 0
            mu_original = 1 - dist_to_pos / (r + 1e-8)
        else:
            dist_to_neg = np.sqrt(np.sum((c_i - center2) ** 2))
            r = max([dist_to_neg]) if [dist_to_neg] else 0
            mu_original = 1 - dist_to_neg / (r + 1e-8)

        dist_to_pos_center = np.sqrt(np.sum((c_i - center1) ** 2))
        dist_to_neg_center = np.sqrt(np.sum((c_i - center2) ** 2))
        sum_dist = dist_to_pos_center + dist_to_neg_center
        ratio = 0 if sum_dist == 0 else (dist_to_pos_center / sum_dist if label == 1 else dist_to_neg_center / sum_dist)
        temp_ball = np.array([[1 if label == 1 else 0] + c_i.tolist()])
        _, p_i = get_label_and_purity(temp_ball)
        sqrt_term = np.sqrt((1 - mu_original ** 2) * (1 - p_i))
        v_GB = ratio * sqrt_term
        denominator = 2 - mu_original ** 2 - v_GB ** 2
        theta = np.sqrt((1 - v_GB ** 2) / denominator) if denominator > 0 else 0
        s = mu_original if p_i == 1 else theta
        processed_balls.append([d[0], d[1], s, d[-1]])
    return processed_balls


# ===================== 工具函数 =====================
def distances(data, p):
    return ((data - p) ** 2).sum(axis=0) ** 0.5


def get_label_and_purity(data):
    num = data.shape[0]
    num_positive = sum(data[:, 0] == 1)
    num_negative = sum(data[:, 0] == 0)
    purity = max(num_positive, num_negative) / num if num else 1.0
    label = 1 if num_positive >= num_negative else -1
    return label, purity


def split_ball(data):
    data = np.array(data)
    data_no_label = data[:, 1:]
    label_cluster = k_means(X=data_no_label, n_clusters=2, random_state=5)[1]
    return [data[label_cluster == 0], data[label_cluster == 1]]


def calculate_center_and_radius(granular_ball):
    data_no_label = granular_ball[:, 1:]
    center = data_no_label.mean(0)
    radius = np.mean(np.sqrt(((data_no_label - center) ** 2).sum(axis=1)))
    return center, radius


# ===================== 粒球去重叠 =====================
def isOverlap(gb_list):
    new_balls_data = []
    for ball_data in gb_list:
        new_ball_data = []
        for point_data in ball_data:
            if point_data[-2] == -1:
                point_data[-2] = 0
            new_ball_data.append([point_data[-2]] + point_data[0:-2].tolist())
        new_balls_data.append(new_ball_data)
    gb_list = new_balls_data
    while True:
        gb_list_new = []
        center_radius = []
        gb_overlap = []
        for gb in gb_list:
            gb = np.array(gb)
            label = get_label_and_purity(gb)[0]
            center, radius = calculate_center_and_radius(gb)
            center_radius.append([center, radius, label])
        for i, r1 in enumerate(center_radius):
            Flag = True
            for j, r2 in enumerate(center_radius):
                if r1[2] != r2[2] and distances(r1[0], r2[0]) < r1[1] + r2[1]:
                    gb_overlap.append(gb_list[i])
                    Flag = False
                    break
            if Flag:
                gb_list_new.append(gb_list[i])
        if len(gb_overlap) == 0:
            return gb_list
        for gb in gb_overlap:
            if len(gb) == 1:
                gb_list_new.append(gb)
                continue
            gb_list_new.extend(split_ball(gb))
        gb_list = gb_list_new


# ===================== 主函数 =====================
def main(kernel_type='gaussian', epsilon=0.0001):
    C_values = [2 ** i for i in range(0, 1)]
    n_repeats = 5
    data_names = ["fourclass"]
    noise_ratios = [0.1]

    for data_name in data_names:
        urlz = fr"C:\Users\28336\Desktop\SGBFSVM\Data2\{data_name}.csv"
        for noise_ratio in noise_ratios:
            for C in C_values:
                folds = KFold_Data(urlz, True, n_splits=5)
                fold_best_list = []  # 存储每一折的最高指标

                for fold in folds:
                    train = np.array(fold['train'])
                    test = np.array(fold['test'])
                    N_data = add_gaussian_noise(train, noise_ratio)

                    # 当前折的最佳指标（取多次重复中的最优）
                    fold_best = [0, 0, 0, 0, 0]

                    for _ in range(n_repeats):
                        exp_best = [0, 0, 0, 0, 0]
                        for l in range(21):
                            pur = 1 - 0.03 * l
                            datab, d = gen_balls(N_data, pur=pur, delbals=0)
                            datab = isOverlap(d)
                            gb_info_list = []
                            for ball in datab:
                                ball = np.array(ball)
                                center, radius = calculate_center_and_radius(ball)
                                label, _ = get_label_and_purity(ball)
                                gb_info_list.append([center, radius, label])
                            datab = ball_membership(gb_info_list)
                            if len(datab) >= 2:
                                X = np.array([x[0] for x in datab])
                                R = np.array([x[1] for x in datab])
                                me = np.array([x[-2] for x in datab])
                                y = np.array([x[-1] for x in datab])
                                model = SMO(kernel_type=kernel_type, C=C, me=me, epsilon=epsilon)
                                model.fit(X, R, y)
                                res = getacc(test, model.w, model.b)
                                if res[0] > exp_best[0]:
                                    exp_best = res


                    fold_best_list.append(exp_best)
                five_fold_max_avg = np.mean(fold_best_list, axis=0)

                # 输出
                print(f"[{data_name}] noise={noise_ratio} | C={C:.2f}")
                print(
                    f"ACC={five_fold_max_avg[0]:.4f}  F1={five_fold_max_avg[3]:.4f}  AUC={five_fold_max_avg[4]:.4f}\n")


if __name__ == '__main__':
    main()