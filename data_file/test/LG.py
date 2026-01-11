import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# 读取文件函数，数据预处理
def read(data):
    # 获取表头（列名）
    columns = data.columns
    y = columns[0]  # 目标变量（因变量）
    x = columns[1:]  # 特征变量（自变量）
    return x, y


def plot_scatter_and_line(ax, x, y, model, xlabel='x_轴', ylabel='y_轴', prediction_data=None):
    """
    绘制散点图和回归线

    参数：
    ax: Matplotlib轴对象，用于绘制图形
    x: 数据的特征（自变量）
    y: 数据的目标值（因变量）
    model: 训练好的模型，用于预测
    xlabel: x轴标签（默认为'人工成本费(元)'）
    ylabel: y轴标签（默认为'产量(公斤)'）
    """
    # 绘制散点图
    ax.scatter(x, y, color='blue', label='实际值')

    # 绘制回归线（通过模型预测）
    ax.plot(x, model.predict(x.reshape(-1, 1)), color='red', label='预测值')

    # 设置坐标轴标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 添加图例
    ax.legend()

    # 如果提供了预测数据，进行预测并输出
    if prediction_data is not None:
        prediction = model.predict(np.array(prediction_data).reshape(-1, 1))
        print(f"预测结果：{prediction}")


# 使用该函数
data = pd.read_excel('./pca2_2.xlsx')
x, y = read(data)

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，或者你可以选择其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

import math
def get_grid_dimensions(n):
    # 计算行数和列数
    rows = math.ceil(n ** 0.5)  # 计算最接近平方根的整数
    cols = math.ceil(n / rows)  # 根据行数计算列数

    # 调整行列数，使得行列的乘积最小且足以容纳所有元素
    while rows * cols < n:
        rows += 1

    return rows, cols
ly, lx = get_grid_dimensions(len(x))
print(lx, ly)
# 创建一个包含多个子图的画布
fig, axs = plt.subplots(lx, ly, figsize=(15, 5))  # len(x) 是列数

for i in range(len(x)):
    current_x = x[i]  # 使用列名
    # 将当前列作为数据 (features)
    x_data = data[current_x].values.reshape(-1, 1)  # 取出自变量数据，并转换为二维数组
    y_data = data[y].values  # 取出因变量数据
    print(len(data[current_x]))  # 检查x_data的样本数
    print(len(data[y]))  # 检查y_data的样本数
    print(x_data.shape)  # 检查 x_data 的形状
    print(y_data.shape)  # 检查 y_data 的形状

    # 初始化模型
    model = LinearRegression()

    # 训练模型
    model.fit(x_data, y_data)

    # 检验线性回归分析模型的拟合程度
    score = model.score(x_data, y_data)
    print(f"模型拟合程度：{score}")
    # print((int)(i / lx),(int)(i % ly))
    if(lx == 1):
        plot_scatter_and_line(axs[i], x_data, y_data, model, xlabel=f'模型 {i+1} 的 x', ylabel=f'模型 {i+1} 的 y')
    else:
        plot_scatter_and_line(axs[(int)(i / lx),(int)(i % ly)], x_data, y_data, model, xlabel=f'模型 {i+1} 的 x', ylabel=f'模型 {i+1} 的 y')
plt.show()
