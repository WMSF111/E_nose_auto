# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 读取文件函数，数据预处理，70%做训练集，30%做预测集
def read(datafile):
    # data = pd.read_csv(datafile) # 读成dataframe格式
    data = pd.read_excel(datafile)
    target = data.iloc[:, 0]  # 第一列类别
    # 将剩余的列作为数据 (features)
    data = data.iloc[:, 1:]  # 所有数据
    # 打乱顺序
    data = data.sample(frac=1).reset_index(drop=True)
    # 创建一个字典，将 target 中的每个元素映射到一个数字
    target_map = make_map(target)
    return target_map, target, data

def make_map(target):
    # 获取类别的排序列表，按照字母或其他顺序排序
    sorted_target = sorted(set(target))  # 获取唯一类别并排序

    # 创建一个字典，将 sorted_target 中的每个元素映射到一个数字，数字从0开始
    target_map = {sorted_target[i]: i for i in range(len(sorted_target))}
    return  target_map

# KNN预测函数
def knn(practice_x, practice_y, predict_x, n_neighbors_num):
    # 测试精度参数为n_neighbors
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors_num, algorithm='auto')
    neigh.fit(practice_x, practice_y)
    predict_y = neigh.predict(predict_x)
    return predict_y


# 将字符类型数据转化为数字类型
def transfer(target_map, array):
    # 使用映射字典转换 array 中的种类标签为数字
    num = [target_map[i] for i in array]
    return num

# 反转映射字典
def reverse_transfer(target_map, array):
    # 创建反向映射字典
    reverse_map = {v: k for k, v in target_map.items()}
    # 使用反向映射字典转换 array 中的数字标签为字符
    char = [reverse_map[i] for i in array]
    return char


# 混淆矩阵画图
def cm_plot(y_test, predict_y, target_map):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    # 生成混淆矩阵
    cm = confusion_matrix(y_test, predict_y)


    # 反转映射字典，将数字标签转换为中文标签
    labels = reverse_transfer(target_map, list(target_map.values()))

    # 绘制混淆矩阵
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，或者你可以选择其他支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

    # 在每个格子内显示数字
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    # 设置 x 和 y 轴的标签为中文标签
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    # 设置轴标签
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')

    return plt


# 读取文件
datafile = './pca2.xlsx'
target_map, target, data = read(datafile) # 训练数据级、预测数据级
num_target = transfer(target_map, target) #转化为数字target
print(data,num_target)

x_train, x_test, y_train, y_test = train_test_split(data, num_target, test_size=0.2, random_state=0, shuffle=True)

# 预测数据的结果和标签，predict_x为数据，predict_y为标签
predict_y = knn(x_train, y_train, x_test, 5)# 预测标签集
print(y_test,predict_y)
cm_plot(y_test,predict_y, target_map).show()
