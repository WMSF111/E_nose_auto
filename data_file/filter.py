# -*- coding: utf-8 -*-
import scipy.signal as signal
import numpy as np
import os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import global_var

script_dir = os.path.dirname(__file__)


def file_get():
    try:
        with open(script_dir + "./resource_ui1/b_0507.txt", 'r') as file:
            text = file.read()

        if len(text) >= 4:
            rows = text.split('\n')  # 每行代表表格中的一行数据
            table_data = [row.split(' ') for row in rows]  # 假设每列用逗号分隔

            num_cols = len(table_data[0])
            data = []
            for i in range(0, len(table_data)):
                if len(table_data[i]) == num_cols:
                    data.append(table_data[i])

            file = pd.DataFrame(data, columns=table_data[0])  # 保存创建的 DataFrame
            return file
        else:
            print("文件内容太短，无法创建 DataFrame。")
            return None
    except FileNotFoundError:
        print("文件未找到。")
        return None
    except Exception as e:
        print("发生错误:", e)
        return None


'''
算术平均滤波法
参数：
    inputs: 输入信号的列表
    window_size: 窗口大小，用于计算中位值，输入整数，越小越接近原数据
返回：
    filtered_output: 滤波后的输出信号列表
'''


# def ArithmeticAverage(inputs,window_size):
#     filtered_output = []
# 	 # 对于每个窗口，计算窗口内数据的平均值并添加到输出列表中
#     for i in range(len(inputs) - window_size + 1):
#         window = inputs[i:i + window_size]
#         window_average = sum(window) / window_size
#         filtered_output.append(window_average)

#     return filtered_output

def ArithmeticAverage(inputs, window_size):
    filtered_output = []

    # 对于每个窗口，计算窗口内数据的平均值并添加到输出列表中
    for i in range(len(inputs)):
        # 计算窗口的开始和结束位置，确保窗口不会超出数据范围
        start = max(i - window_size // 2, 0)
        end = min(i + window_size // 2 + 1, len(inputs))
        window = inputs[start:end]
        window_average = sum(window) / len(window)
        # filtered_output.append((int)(window_average))
        filtered_output.append((window_average))

    return filtered_output


'''
递推平均滤波法
参数：
    inputs: 输入信号的列表
    window_size: 窗口大小，用于计算中位值，输入整数，越小越接近原数据
返回：
    filtered_output: 滤波后的输出信号列表
'''


# def SlidingAverage(inputs,window_size):
#     filtered_output = []
#     window_sum = sum(inputs[:window_size])  # 初始窗口内数据的总和

#     # 初始窗口的平均值作为第一个输出
#     filtered_output.append(window_sum / window_size)

#     # 对于每个后续的数据点，利用递推公式更新窗口内数据的总和，并计算平均值
#     for i in range(window_size, len(inputs)):
#         window_sum = window_sum - inputs[i - window_size] + inputs[i]  # 更新窗口内数据的总和
#         filtered_output.append(window_sum / window_size)  # 计算平均值并添加到输出列表中

#     return filtered_output

def SlidingAverage(inputs, window_size):
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if len(inputs) < window_size:
        raise ValueError("Input length must be at least as large as the window size.")

    filtered_output = []
    window_sum = sum(inputs[:window_size])  # 初始窗口内数据的总和

    # 初始窗口的平均值作为第一个输出
    # filtered_output.append((int)(window_sum / window_size))
    filtered_output.append((window_sum / window_size))
    # 对于每个后续的数据点，利用递推公式更新窗口内数据的总和，并计算平均值
    for i in range(window_size, len(inputs)):
        window_sum = window_sum - inputs[i - window_size] + inputs[i]  # 更新窗口内数据的总和
        # filtered_output.append((int)(window_sum / window_size))  # 计算平均值并添加到输出列表中
        filtered_output.append(window_sum / window_size)  # 计算平均值并添加到输出列表中

    # 计算填充值
    padding_size = window_size // 2

    # 前端填充
    front_padding = [filtered_output[0]] * padding_size
    # 后端填充
    back_padding = [filtered_output[-1]] * (window_size - 1 - padding_size
                                            )
    # 合并填充
    filtered_output = front_padding + filtered_output + back_padding

    return filtered_output[:len(inputs)]  # 确保最终输出长度与输入长度一致


'''
中位值平均滤波法
参数：
    inputs: 输入信号的列表
    window_size: 窗口大小，用于计算中位值，输入整数，越小越接近原数据
返回：
    filtered_output: 滤波后的输出信号列表
'''


def MedianAverage(inputs, window_size):
    filtered_output = []

    # 边缘情况处理：处理前 window_size//2 个值
    for i in range(window_size // 2):
        filtered_output.append(inputs[i])

    # 对于每个窗口，计算中位值并将其添加到结果列表中
    for i in range(len(inputs) - window_size + 1):
        window = inputs[i:i + window_size]
        # median_value = (int)(np.median(window))
        median_value = np.median(window)
        filtered_output.append(median_value)

    # 边缘情况处理：处理后 window_size//2 个值
    for i in range(len(inputs) - window_size + 1, len(inputs)):
        filtered_output.append(inputs[i])

    # 处理边缘情况以确保输出长度与输入长度一致
    if len(filtered_output) < len(inputs):
        # 如果输出长度少于输入长度, 复制最后一个值
        filtered_output.extend([filtered_output[-1]] * (len(inputs) - len(filtered_output)))
    elif len(filtered_output) > len(inputs):
        # 如果输出长度多于输入长度, 剪切输出列表
        filtered_output = filtered_output[:len(inputs)]

    return filtered_output


'''
一阶滞后滤波法
    方法：本次滤波结果=（1-a）*本次采样值+a*上次滤波结果。 
a:滞后程度决定因子，0~1（越大越接近原数据）
'''


# def FirstOrderLag(inputs,a):

# 	filtered_output = inputs

# 	tmpnum = inputs[0]							#上一次滤波结果
# 	for index,tmp in enumerate(inputs):
# 		filtered_output[index] = (int)((1-a)*tmp + a*tmpnum)
# 		tmpnum = tmp
# 	return filtered_output
def FirstOrderLag(inputs, a):
    if not (0 < a < 1):
        raise ValueError("a should be between 0 and 1 (exclusive).")

    filtered_output = [0] * len(inputs)  # 初始化过滤输出列表
    if len(inputs) == 0:
        return filtered_output

    tmpnum = inputs[0]  # 上一次滤波结果

    # 遍历输入列表并进行滤波
    for index, tmp in enumerate(inputs):
        # filtered_output[index] = (int)((1 - a) * tmp + a * tmpnum)
        filtered_output[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = filtered_output[index]  # 更新上一次滤波结果为当前滤波结果

    return filtered_output


'''
    加权递推平均滤波法
    参数：
    inputs: 输入信号的列表
    alpha: 平滑系数，范围在0到1之间（越大越接近原数据）
    返回：
    filtered_output: 滤波后的输出信号列表
'''


# def WeightBackstepAverage(inputs, alpha):

#     filtered_output = []
#     filtered_output.append(inputs[0])  # 初始时，输出等于第一个输入值

#     for i in range(1, len(inputs)):
#         # 递推式：滤波后的当前值等于上一个滤波后的值乘以(1-alpha)，再加上当前输入值乘以alpha
#         filtered_value = (int)((1 - alpha) * filtered_output[-1] + alpha * inputs[i])
#         filtered_output.append(filtered_value)

#     return filtered_output

def WeightBackstepAverage(inputs, alpha):
    if not (0 < alpha < 1):
        raise ValueError("alpha should be between 0 and 1 (exclusive).")

    filtered_output = [0] * len(inputs)
    if len(inputs) == 0:
        return filtered_output

    filtered_output[0] = inputs[0]  # 初始时，输出等于第一个输入值

    for i in range(1, len(inputs)):
        # 递推式：滤波后的当前值等于上一个滤波后的值乘以(1-alpha)，再加上当前输入值乘以alpha
        # filtered_output[i] = (int)((1 - alpha) * filtered_output[i - 1] + alpha * inputs[i])
        filtered_output[i] = (1 - alpha) * filtered_output[i - 1] + alpha * inputs[i]

    return filtered_output


'''
消抖滤波法
    检查是否有连续N个不相同的元素，如果有，则将后续的元素设置为与这N个元素的第一个值。
N:消抖上限,范围在2以上。
'''


# def ShakeOff(inputs,N):

# 	filtered_output = inputs

# 	usenum = filtered_output[0]								#有效值
# 	i = 0 											#标记计数器
# 	for index,tmp in enumerate(filtered_output):
# 		if tmp != usenum:
# 			usenum = filtered_output[index - 1]
# 			i = i + 1
# 			if i >= N:
# 				i = 0
# 				filtered_output[index] = usenum
# 		else:
# 			i = 0
# 	return filtered_output

def ShakeOff(inputs, N):
    if N <= 0:
        raise ValueError("N must be greater than 0.")

    filtered_output = inputs[:]  # 复制输入数据以避免直接修改
    if len(inputs) == 0:
        return filtered_output

    usenum = filtered_output[0]  # 有效值
    i = 0  # 标记计数器

    for index in range(1, len(filtered_output)):
        tmp = filtered_output[index]
        if tmp != usenum:
            usenum = filtered_output[index - 1]
            i += 1
            if i >= N:
                i = 0
                filtered_output[index] = usenum
        else:
            i = 0

    return filtered_output


'''
限幅消抖滤波法
    首先，它检查相邻元素之间的差值是否超过了给定的振幅（Amplitude），
	如果超过，则将该元素的值设为前一个元素的值。
	然后，它检查是否有连续N个相同的元素，如果有，则将后续的元素设置为与这N个元素相同的值。
Amplitude:限制最大振幅,范围在0 ~ ∞ 建议设大一点
N:消抖上限,范围在0 ~ ∞
'''


def AmplitudeLimitingShakeOff(inputs, Amplitude, N):
    filtered_output = inputs.copy()

    tmpnum = filtered_output[0]
    for index, newtmp in enumerate(filtered_output):
        if np.abs(tmpnum - newtmp) > Amplitude:
            filtered_output[index] = tmpnum
        tmpnum = newtmp

    usenum = filtered_output[0]
    i = 0
    for index2, tmp2 in enumerate(filtered_output):
        if tmp2 != usenum:
            usenum = tmp2
            i += 1
            if i >= N:
                i = 0
                filtered_output[index2] = usenum
        else:
            i = 0

    return filtered_output

# # File = pd.read_csv(os.path.join(str(script_dir), 'train', 'trainfile.csv'), header=None, skiprows=1) #读取文件,得到dataframe数据
# # File = file_get()
# file_path = os.path.join(str(script_dir), 'train', 'trainfile.csv')
# File = pd.read_csv(file_path, skiprows=1, usecols=range(1, pd.read_csv(file_path, nrows=1).shape[1]))
# # 重新设置 DataFrame 的索引
# File.reset_index(drop=True, inplace=True)
# print(File)
# train_data = np.array(File) #先将数据框转换为数组
# train_data_list = train_data.tolist()  #其次转换为列表
# T = np.arange(0, 0.5, 1/4410.0)
# num = signal.chirp(T, f0=10, t1 = 0.5, f1=1000.0)
# # print(num)
# File2 = File
# # print(File2)
# for column in File:
#       column_data = File[column].astype(int)
#       column_data = column_data.tolist()
#     #   pl.subplot(2,1,1)
#     #   pl.plot(column_data)
#       result = SlidingAverage(column_data.copy(), 2)
#       print(len(result))
#       #print(num - result
#     #   pl.subplot(2,1,2)
#     #   pl.plot(result)
#     #   pl.show()
#       File2[column] = column_data
# print(File2)

