import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import global_var as global_var

# 降维模型
from sklearn.decomposition import PCA
# LDA与分类相同模块，降维时设置n_components参数
from sklearn.manifold import LocallyLinearEmbedding  # LLE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

class Dimension:
    """
    降维模型：PCA、LDA(线性判别分析)、LLE（局部线性嵌入）
    """
    def __init__(self, DataFrame, test_size, random_state=42, mix=True):
        """ LDA算法模型
            Args:
                DataFrame:  数据集
                test_size:  测试集占总数据集的比例
                random_state: 数据分割的随机性种子（random_state=42 结果可复现， random_state=None 每次运行结果不同）
                Mix： True打乱顺序
        """
        self.target_map, self.target, self.data = read(DataFrame, mix)  # 训练数据级、预测数据级
        self.num_target = transfer(self.target_map, self.target)  # 转化为数字target

        self.random_state = random_state

        # stratify-保持分割后训练集和测试集中类别的比例与原数据集一致(这里取y，表示按标签的分布进行分层抽样)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state, stratify=self.target)
        print(f"训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")

        self.X_train, self.X_test = standart(self.X_train, self.X_test)     # 标准化训练集和测试集
        self.X_combine = np.vstack((self.X_train, self.X_test))        # 竖直堆叠数组
        self.y_combine = np.hstack((self.y_train, self.y_test))        # 水平拼接数组

    def pca(self, test_size, kernel_str, C):
        """ PCA 主成分分析 """

        print("分类模型PCA......")

        # 定义不同维度的PCA
        n_components_list = [2, 5, 10, 20, self.data.shape[1]]  # 包括原始特征

        results = []

        for n_components in n_components_list:
            if n_components > self.data.shape[1]:
                continue

            # 创建管道
            if n_components == self.data.shape[1]:
                # 使用所有特征（无PCA）
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(random_state=self.random_state))
                ])
                method = '原始特征'
            else:
                # 使用PCA降维
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components, random_state=42)),
                    ('classifier', RandomForestClassifier(random_state=42))
                ])
                method = f'PCA({n_components})'

            pipeline.fit(self.X_train, self.y_train)         # 训练
            y_pred = pipeline.predict(self.X_test)           # 预测
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"准确率: {accuracy:.4f}")

            # 如果是PCA，计算解释方差
            if n_components != self.data.shape[1]:
                pca = pipeline.named_steps['pca']
                explained_variance = pca.explained_variance_ratio_.sum()
            else:
                explained_variance = 1.0

            results.append({
                '方法': method,
                '特征数': n_components,
                '准确率': accuracy,
                '解释方差': explained_variance
            })

        return results    #返回

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

def standart(x_train, x_test,):
    # ? 标准化训练集和测试集
    sc = StandardScaler()  # 定义一个标准缩放器
    sc.fit(x_train)  # 计算均值、标准差
    x_train = sc.transform(x_train)  # 使用计算出的均值和标准差进行标准化
    x_test = sc.transform(x_test)  # 使用计算出的均值和标准差进行标准化

    return x_train, x_test



class DIMENSION():
    def __init__(self, ui, textEdit_DataFrame):
        self.ui = ui
        self.df = textEdit_DataFrame  # 读取文件
        self.data = pd.DataFrame(self.df.iloc[:, 1:]) # 除第一行外的所有数据

    def OLS(self, data):
        cols = global_var.headers[1:]
        # print(cols)
        for i in range(0, len(cols)):
            data1 = data[cols]
            x = sm.add_constant(data1)  # 生成自变量
            y = data['target']  # 生成因变量
            model = sm.OLS(y, x)  # 生成模型
            result = model.fit()  # 模型拟合
            print(result.summary())  # 模型描述
            pvalues = result.pvalues  # 得到结果中所有P值
            pvalues.drop('const', inplace=True)  # 把const取得
            pmax = max(pvalues)  # 选出最大的P值
            if pmax > 0.05:
                ind = pvalues.idxmax()  # 找出最大P值的index
                cols.remove(ind)  # 把这个index从cols中删除
            else:
                self.finalData = result
                break
    def pca(self, num,  target_variance=80):
        target_variance = target_variance / 100
        # 标准化数据
        scaler = StandardScaler()

        data_scaled = scaler.fit_transform(self.data)

        # 应用 PCA
        pca_f = PCA(n_components=num, svd_solver="full")
        pca_f = pca_f.fit(data_scaled)

        self.finalData = pca_f.transform(data_scaled)
        df = pd.DataFrame(self.finalData) # 转化成DataFrame
        df.insert(0, self.df.columns[0], self.df[self.df.columns[0]].values)# 将 new_column 添加到 DataFrame 的第一列


        variance_ratios = pca_f.explained_variance_ratio_  # 贡献率

        cumulative_variance_ratios = np.cumsum(variance_ratios) # 计算累积贡献率
        # 将累积贡献率转换为字符串并追加到 QTextBrowser
        cumulative_variance_ratios_str = ', '.join(map(str, cumulative_variance_ratios))

        self.ui.textBrowser.clear()
        # 将主成分个数转换为字符串并追加到 QTextBrowser
        self.ui.textBrowser.append('主成分个数: ' + str(len(variance_ratios)))
        self.ui.textBrowser.append('累计贡献率：' + cumulative_variance_ratios_str)
        if cumulative_variance_ratios[-1] > target_variance:
            # 找到累积贡献率达到目标值的最小主成分数
            num_components = np.argmax(cumulative_variance_ratios >= target_variance) + 1
            # 追加需要的主成分数和累积贡献率
            self.ui.textBrowser.append(
                f"\n需要 {num_components} 个主成分才能达到累积贡献率 {target_variance * 100:.2f}% 以上。")
        else:
            self.ui.textBrowser.append(
                f"\n {num} 个主成分无法达到累积贡献率 {target_variance * 100:.2f}% 以上，请增加维度尝试，或者切换算法。")
        return df, variance_ratios


    def lda(self, num, target_variance):
        lda_model = LinearDiscriminantAnalysis(n_components=num)
        self.finalData = lda_model.fit_transform(self.data, self.df[self.df.columns[0]].values)
        df = pd.DataFrame(self.finalData)  # 转化成DataFrame
        df.insert(0, self.df.columns[0], self.df[self.df.columns[0]].values)# 将 new_column 添加到 DataFrame 的第一列
        # self.finalData = df.to_numpy()  # 如果需要，你可以将 df 转回 NumPy 数组

        # 获取解释方差比
        explained_variance_ratio = lda_model.explained_variance_ratio_
        sum_explained_variance_ratio = np.cumsum(explained_variance_ratio)
        # 将数据转换到投影空间
        transformed_data = lda_model.transform(self.data)
        # 计算类内散布矩阵
        Sw = np.cov(transformed_data.T)
        #    计算类间散布矩阵
        means_diff = (lda_model.means_[1] - lda_model.means_[0]).reshape(-1, 1)
        SB = np.outer(means_diff, means_diff)
        # 计算类间散布与类内散布之比
        inter_intra_ratio = np.trace(SB) / np.trace(Sw)

        self.ui.textBrowser.clear()
        # 将主成分个数转换为字符串并追加到 QTextBrowser
        self.ui.textBrowser.append('解释方差比: ' + str(explained_variance_ratio))
        self.ui.textBrowser.append("解释方差比之和: " + str(sum_explained_variance_ratio))
        self.ui.textBrowser.append("类间散布与类内散布之比: " + str(inter_intra_ratio))
        if sum_explained_variance_ratio[-1] * 100 > target_variance :
            # 找到累积贡献率达到目标值的最小主成分数
            num_components = np.argmax(sum_explained_variance_ratio >= target_variance) + 1
            # 追加需要的主成分数和累积贡献率
            self.ui.textBrowser.append(
                f"\n需要 {num_components} 个主成分才能达到累积贡献率 {target_variance:.2f}% 以上。")
        else:
            self.ui.textBrowser.append(
                f"\n {num} 个主成分无法达到累积贡献率 {target_variance:.2f}% 以上，请增加维度尝试，或者切换算法。")

        return df, explained_variance_ratio # dataframe格式返回
