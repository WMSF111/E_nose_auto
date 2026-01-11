# -*- coding: utf-8 -*-
# 只能处理2维数据
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from svm_smo_classifier import SVMClassifier


# 读取文件函数，数据预处理，70%做训练集，30%做预测集
def read(datafile):
    data = pd.read_excel(datafile)
    # data = pd.read_csv(datafile)
    target = data.iloc[:, 0] #第一列类别
    # 将剩余的列作为数据 (features)
    data = data.iloc[:, 1:] #所有数据
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
    return target_map

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
def cm_plot(t1, output):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(t1, output)
    #    print(cm)
    #    print(len(cm))
    import matplotlib.pyplot as plt
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    return plt


# 读取文件
datafile = '../pca2.xlsx'
target_map, target, data = read(datafile) # 训练数据级、预测数据级
num_target = transfer(target_map, target)

x_train, x_test, y_train, y_test = train_test_split(data, num_target, test_size=0.2, random_state=0, shuffle=True)
# C = 1000，倾向于硬间隔
svm = SVMClassifier(C=1000)
svm.fit(x_train, y_train)
y_test_pred = svm.predict(x_test)
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(14, 10))
plt.subplot(221)
svm.plt_svm(x_train, y_train, is_show=False, is_margin=True)
plt.subplot(222)
svm.plt_loss_curve(is_show=False)

# C = 1，倾向于软间隔
svm = SVMClassifier(C=1)
svm.fit(x_train, y_train)
y_test_pred = svm.predict(x_test)
print(classification_report(y_test, y_test_pred))

plt.subplot(223)
svm.plt_svm(x_train, y_train, is_show=False, is_margin=True)
plt.subplot(224)
svm.plt_loss_curve(is_show=False)

plt.tight_layout()
plt.show()
