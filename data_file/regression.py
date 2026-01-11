import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class TRAIN():
    # 输入： finalData=数据级 target =数据标签 test_size = 训练集数量（0.1-0.9）
    def __init__(self, ui, data):
        self.ui = ui
        # 获取表头（列名）
        columns = data.columns
        self.y = columns[0]  # 目标变量（因变量）
        self.x = columns[1:]  # 特征变量（自变量）

    def LG(self, x, y):
        # 初始化模型
        model = LinearRegression()
        # 训练模型
        model.fit(x, y)
        # 检验线性回归分析模型的拟合程度
        score = model.score(x, y)
        return model,score


# 读取文件函数，数据预处理，70%做训练集，30%做预测集
def read(data, Mix):
    target = data.iloc[:, 0]  # 第一列类别
    # 将剩余的列作为数据 (features)
    data = data.iloc[:, 1:]  # 所有数据
    if Mix == True:
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

# 将字符类型数据转化为数字类型
def transfer(target_map, array):
    # 使用映射字典转换 array 中的种类标签为数字
    num = [target_map[i] for i in array]
    return num