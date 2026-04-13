import csv
import time
import pandas as pd
import numpy as np
from sklearn.cluster import k_means,KMeans
import json
import csv
# import sklearn.cluster.k_means_ as KMeans
from sklearn.cluster import KMeans
import warnings
from collections import Counter
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

warnings.filterwarnings("ignore")  # 处理警告



class GranularBall:
    """class of the granular ball"""
    def __init__(self, data):
        """
        :param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
        and each of the preceding columns corresponds to a feature
        """
        self.r_mode = r_mode
        self.clf = self.get_clf()
        self.data = data[:, :]
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.label, self.purity ,self.r= self.__get_label_and_purity_and_r()

    def __get_label_and_purity_and_r(self):
        """
        :return: the label and purity of the granular ball.
        """
        count = Counter(self.data[:, -2])
        label = max(count, key=count.get)
        purity = count[label] / self.num
        arr=np.array(self.data_no_label)-self.center
        ar=np.square(arr)
        a=np.sqrt(np.sum(ar,1))
        r=np.sum(a)/len(self.data_no_label)
        return label, purity,r

    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        """
        # label_cluster = KMeans(X=self.data_no_label, n_clusters=2)[1]
        clu=KMeans(n_clusters=2).fit(self.data_no_label)
        label_cluster = clu.labels_
        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :])
            ball2 = GranularBall(self.data[label_cluster == 1, :])
        else:
            ball1 = GranularBall(self.data[0:1, :])
            ball2 = GranularBall(self.data[1:, :])
        return ball1, ball2

class GBList:
    """class of the list of granular ball"""
    def __init__(self, data=None):
        self.data = data[:, :]
        self.granular_balls = [GranularBall(self.data)]  # gbs is initialized with all data

    def init_granular_balls(self, purity=1, min_sample=1):
        """
        Split the balls, initialize the balls list.
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        ll = len(self.granular_balls)
        i = 0
        while True:
            # print(self.granular_balls[i].purity, purity)
            if (self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample) or (len(self.granular_balls) <= 2 and self.granular_balls[i].num >= 2):
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_r(self):
        """
        :return: 返回半径r
        """
        return np.array(list(map(lambda x: x.r, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)
    def del_ball(self,purty=0.,num_data=0):
        T_ball=[]
        for ball in self.granular_balls:
            if ball.purity >= purty and ball.num >= num_data:
                T_ball.append(ball)
        self.granular_balls=T_ball.copy()
        self.data=self.get_data()
    # def re_division(self, i):
    #     """
    #     Data division with the center of the ball.
    #     :return: a list of new granular balls after divisions.
    #     """
    #     k = len(self.granular_balls)
    #     attributes = list(range(self.data.shape[1] - 2))
    #     attributes.remove(i)
    #     clu = KMeans(n_clusters=k, init=self.get_center()[:, attributes], max_iter=1).fit(self.data[:, attributes])
    #     label_cluster = clu.labels_
    #     # label_cluster = KMeans(X=self.data[:, attributes], n_clusters=k,
    #     #                         init=self.get_center()[:, attributes], max_iter=1)[1]
    #     granular_balls_division = []
    #     for i in set(label_cluster):
    #         granular_balls_division.append(GranularBall(self.data[label_cluster == i, :]))
    #     return granular_balls_division
def generate_ball_data(data,pur,delbals):
    num, dim = data[:, :-1].shape
    index = np.array(range(num)).reshape(num, 1)  # column of index
    data = np.hstack((data, index))  # Add the index column to the last column of the data
    # step 1.
    #print(data[0:4])
    gb = GBList(data)  # create the list of granular balls
    gb.init_granular_balls(purity=pur)  # initialize the list
    # print(len(gb.granular_balls))
    gb.del_ball(num_data=delbals)

    centers=gb.get_center().tolist()
    rs=gb.get_r().tolist()
    # print(type(centers[0]))
    balldata = []  # 检验
    for i in range(len(gb.granular_balls)):
        a=[]
        a.append(centers[i])
        a.append(rs[i])
        # print(data[i][-2])
        if gb.granular_balls[i].label==-1:
            a.append(-1)
        elif gb.granular_balls[i].label==1:
            a.append(1)
        balldata.append(a)
    # print(balldata[0])
    return balldata

def split_ball(data, splitting_method):
    # 临时去除数据标签
    data_no_label = data[:, 1:]
    k = 2
    if splitting_method == 'k-means':
        # X: 数据; n_clusters: K的值; random_state: 随机状态（为了保证程序每次运行都分割一样的训练集和测试集）
        # 初始中心选取默认采用Kmeans++, 选取思想是聚类中心互相离得越远越好
        label_cluster = k_means(X=data_no_label, n_clusters=k, random_state=5)[1]  # 返回划分后的聚类标签，顺序和原始输入数据顺序一致
        # kmeans = KMeans(n_clusters=k, random_state=5).fit(data_no_label)
        # label_cluster =  kmeans.labels_   # 返回划分后的聚类标签，顺序和原始输入数据顺序一致
        # tmp = np.sum(label_cluster2==label_cluster)
        pass
    elif splitting_method == 'center_split':
        # 采用正、负类中心直接划分
        p_left = data[data[:, 0] == 1, 1:].mean(0)
        p_right = data[data[:, 0] == 0, 1:].mean(0)
        distances_to_p_left = distances(data_no_label, p_left)
        distances_to_p_right = distances(data_no_label, p_right)
        relative_distances = distances_to_p_left - distances_to_p_right
        label_cluster = np.array(list(map(lambda x: 0 if x <= 0 else 1, relative_distances)))
    elif splitting_method == 'center_means':
        # 采用正负类中心作为 2-means 的初始中心点
        p_left = data[data[:, 0] == 1, 1:].mean(0)
        p_right = data[data[:, 0] == 0, 1:].mean(0)
        centers = np.vstack([p_left, p_right])
        label_cluster = k_means(X=data_no_label, n_clusters=2, init=centers, n_init=1)[1]
    else:
        return data
    # 根据聚类标签，将原始输入数据划分为两簇，即为两个粒球
    ball1 = data[label_cluster == 0, :]
    ball2 = data[label_cluster == 1, :]
    #plot_gb([ball1,ball2],0,'asd')
    return [ball1, ball2]

def distances(x, c):
    return np.sqrt(np.sum(np.square(x - c)))

def deoverlap(GB_list=[],clf='',r_mode='',purity=0):
    GB_minsize = 2
    sample_num_raw = sum([x.data.shape[0] for x in GB_list])
    positive_list = [x for x in GB_list if x.lable==1]
    negative_list = [x for x in GB_list if x.lable!=1]
    #分解positive球
    while True:
        split_list = []
        leave_list = []
        for gb_p in positive_list:
            for gb_n in negative_list:
                # if gb_p.data.shape[0] <= GB_minsize:
                #     break
                if distances(gb_p.center, gb_n.center) < gb_p.radius + gb_n.radius: #and gb_p.purity!=1:
                    split_list.append(gb_p)
                    split_list.append(gb_n)
                    break
        split_list = list(set(split_list))
        if len(split_list) == 0:
            break
        else:
            #print("split len:",len(split_list))
            pass
        leave_list = [x for x in positive_list if x not in split_list]
        leave_list += [x for x in negative_list if x not in split_list]
        leave_list = [x for x in leave_list if x.data.shape[0]>=GB_minsize]
        tmp_list = []
        for gb in split_list:
            if gb.data.shape[0]>=2:#至少有2个样本才能split
                tmp_list.extend(split_ball(gb.data, "k-means"))
        tmp_list = [x for x in tmp_list if x.data.shape[0] >= GB_minsize]#球至少有2个样本
        tmp_list = [GB(x,clf,r_mode,purity) for x in tmp_list ]
        #print("tmp len:",len(tmp_list))

        total_list = leave_list + tmp_list
        positive_list = [x for x in total_list if x.lable == 1]
        negative_list = [x for x in total_list if x.lable != 1]
        sample_num = sum([x.data.shape[0] for x in total_list])
        #print("{}     {}",sample_num_raw,sample_num)
    return positive_list + negative_list



def gen_balls(data,pur,delbals):
    # df=pd.read_csv(url,header=None)
    # data=df.values
    # print(data.shape)
    #print(data[0],"data0")
    balls=generate_ball_data(data,pur=pur,delbals=delbals)
    R_balls=[]
    for i in balls:
        t_ball=[]
        t_ball.append(i[0])
        t_ball.append(i[1])
        t_ball.append(i[2])
        R_balls.append(t_ball)
    print("粒球数量：",len(balls))
    return balls
    """
    with open(savurl, "w", newline='', encoding="utf-8") as jg:
        writ = csv.writer(jg)
        for i in balls:
            writ.writerow([i[0],i[1],i[2]])
url="D:\\py\\粒球SVM_精度\噪声数据集\\sonartrainN0.1.csv"
pur=0.94
delbals=3
savurl="D:\\py\\粒球SVM_精度\粒球数据\\球sonartrainN0.1.csv"
gen_balls(url,pur,delbals,savurl)
"""