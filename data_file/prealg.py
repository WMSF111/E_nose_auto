import pandas as pd
import data_file.filter as filter

class Pre_Alg():
    def __init__(self, ui, textEdit_DataFrame, select):
        self.ui = ui
        self.select = select
        self.type = float
        self.data = textEdit_DataFrame.iloc[:, 1:].copy()  # 去除列头行头
        self.result = textEdit_DataFrame.copy()

    def Filter_Choose(self):
        for column in self.data.columns:
            column_data = self.data[column].to_numpy() # 转化为数组
            if self.select == "算术平均滤波法":
                # window_size: 窗口大小，用于计算中位值，输入整数，越小越接近原数据
                result = filter.ArithmeticAverage(column_data.copy(), 2)
                # window_size: 窗口大小，用于计算中位值，输入整数，越小越接近原数据
            elif self.select == "递推平均滤波法":
                result = filter.SlidingAverage(column_data.copy(), 2)
            elif self.select == "中位值平均滤波法":
                result = filter.MedianAverage(column_data.copy(), 2)
            elif self.select == "一阶滞后滤波法":
                # 滞后程度决定因子，0~1（越大越接近原数据）
                result = filter.FirstOrderLag(column_data.copy(), 0.9)
            elif self.select == "加权递推平均滤波法":
                # 平滑系数，范围在0到1之间（越大越接近原数据）
                result = filter.WeightBackstepAverage(column_data.copy(), 0.9)
            elif self.select == "消抖滤波法":
                # N:消抖上限,范围在2以上。
                result = filter.ShakeOff(column_data.copy(), 4)
            elif self.select == "限幅消抖滤波法":
                # Amplitude:限制最大振幅,范围在0 ~ ∞ 建议设大一点
                # N:消抖上限,范围在0 ~ ∞
                result = filter.AmplitudeLimitingShakeOff(column_data.copy(), 200, 3)
            # 将 result 列表中的元素显式转换为目标列的数据类型
            result = [self.result[column].dtype.type(value) for value in result]
            # 将处理后的数据直接更新回原始的 DataFrame
            self.result.iloc[:, self.result.columns.get_loc(column)] = result
        return self.result

    def Val_Choose(self): # 返回Dataframe
        self.result.set_index(self.result.columns[0], inplace=True)#第一列作为索引
        self.result = self.result.apply(pd.to_numeric, errors='coerce') # 全部转化为数字
        if(self.select == "平均值"):
            df_new = self.result.groupby(self.result.index).median() # 按平均值新建dataframe
        elif(self.select == "中位数"):
            df_new = self.result.groupby(self.result.index).mean()
        elif(self.select == "众数"):
            # mode()，它会返回一个 DataFrame 或者 Series，包含每个组的众数（如果有多个众数，它会列出所有的众数）
            # iloc[0] 用来选取第一个众数（如果有多个众数）
            df_new = (self.result.groupby(self.result.index).
                      agg(lambda x: x.mode().iloc[0]))
        elif self.select == "极差":
            df_new = self.result.groupby(self.result.index).agg(lambda x: x.max() - x.min())
        elif self.select == "最大值":
            df_new = self.result.groupby(self.result.index).agg(lambda x: x.max())
        elif self.select == "最大斜率":
            # 计算每组相邻数据点的斜率
            df_new = self.result.groupby(self.result.index).apply(
                self.calculate_slope
            )
        # 确保返回的 DataFrame 与原始数据类型一致
        df_new = df_new.astype(self.result.iloc[:, 0].dtype)

        # 取消索引，将 'target' 列恢复为普通列
        df_new.reset_index(inplace=True)
        return df_new

    # 定义计算斜率的函数
    def calculate_slope(self, group):
        """
        计算每组相邻数据点的斜率
        group: DataFrame，包含一组数据
        返回：一个包含斜率的 Series
        """
        # 计算相邻数据点的斜率
        print(group)
        # 假设 df 是你的原始数据
        # 计算每列的差值
        slope_df = group.diff()  # 计算相邻行之间的差值

        # 计算每列的最大斜率
        max_slope = slope_df.max()  # 找到每列的最大差值，即最大斜率
        # 显示每列的最大斜率
        print(max_slope)
        return max_slope


