'''算法显示'''
from resource_ui.alg_puifile.Predict_uishow import Ui_Pre_show
from resource_ui.alg_puifile.Dimension_uishow import Ui_Dimension_show
from resource_ui.alg_puifile.SVM_uishow import Ui_SVM_UI # 导入生成的 UI 类
from resource_ui.alg_puifile.KNN_uishow import Ui_Form as Ui_KNN_UI # 导入生成的 UI 类
from resource_ui.alg_puifile.Regression_uishow import Ui_Regression as Regression_uishow  # 导入生成的 UI 类
from PySide6.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout # 改为继承 QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import global_var as global_var
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
import os
from resource_ui.modules import *

def type_set(a):
    useCustomTheme = True
    themeFile = global_var.themeFile

    if useCustomTheme:
        with open(themeFile, 'r') as file:
            theme = file.read()  # 读取QSS文件内容
            a.setStyleSheet(theme)  # 将样式应用到当前窗口
class ADDTAB():
    def __init__(self, The_QTabWidget):
        self.The_QTabWidget = The_QTabWidget

    # 新建新的绘图TAB
    def add_plot_tab(self, Dnum=2, title="Plot", plot_function=None, plot_function2=None, *args, **kwargs):
        new_tab = QWidget()

        # 创建主垂直布局
        layout = QVBoxLayout(new_tab)

        # 创建一个垂直布局来放置图表
        horizontal_layout = QVBoxLayout()

        # 创建第一个Matplotlib图表
        fig1 = Figure(figsize=(6, 4))
        canvas1 = FigureCanvas(fig1)
        if Dnum == 3:
            ax1 = fig1.add_subplot(111, projection='3d')
        else:
            ax1 = fig1.add_subplot(111)

        # 如果提供了绘图函数，调用它
        if plot_function:
            plot_function(ax1, *args, **kwargs)

        # 将第一个图表添加到水平布局
        horizontal_layout.addWidget(canvas1)

        # 如果需要创建第二个图表
        if plot_function2 != None:
            # 创建第二个Matplotlib图表
            fig2 = Figure(figsize=(6, 4))
            canvas2 = FigureCanvas(fig2)
            if Dnum == 3:
                ax2 = fig2.add_subplot(111, projection='3d')
            else:
                ax2 = fig2.add_subplot(111)

            # 如果提供了绘图函数，调用它
            if plot_function2:
                plot_function2(ax2, *args, **kwargs)

            # 将第二个图表添加到水平布局
            horizontal_layout.addWidget(canvas2)

        # 将水平布局添加到主垂直布局中
        layout.addLayout(horizontal_layout)

        # 添加新的 Tab 到 QTabWidget
        self.The_QTabWidget.addTab(new_tab, title)

        # 切换到新添加的 Tab
        self.The_QTabWidget.setCurrentIndex(self.The_QTabWidget.count() - 1)


    # 创建一个新的文本Tab
    def add_text_tab(self, finaldata, title="Plot", html = False):
        showdata = finaldata
        if html == True:
            showdata = finaldata.to_html(index=False)  # 转换为HTML表格格式
        new_tab = QWidget()
        layout = QVBoxLayout(new_tab)
        new_textedit = QTextEdit()
        # new_textedit.setFontFamily("Consolas")  # 设置为固定宽度字体（Courier）
        # 将图表添加到布局
        layout.addWidget(new_textedit)
        title = title

        new_textedit.append(showdata)

        # 添加新的 Tab 到 QTabWidget
        self.The_QTabWidget.addTab(new_tab, title)
        self.The_QTabWidget.setCurrentIndex(self.The_QTabWidget.count() - 1)  # 切换到新添加的 Tab
        # 设置保存快捷键
        self.set_save_shortcut(new_tab, finaldata)

    def set_save_shortcut(self, new_tab, text):
        """为新 Tab 设置保存快捷键"""
        # 创建一个 QAction
        save_action = QAction("Save", self.The_QTabWidget)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(lambda: ADDTAB.save_text(text))

        # 将 QAction 添加到新 Tab 的上下文菜单
        new_tab.addAction(save_action)

    def save_text(text):
        """保存当前文本"""
        # 确保text是DataFrame
        if isinstance(text, str):  # 如果传入的是字符串，尝试转回DataFrame
            text = pd.read_csv(StringIO(text))  # 从文本恢复DataFrame
        # 设置文件类型过滤器，增加xlsx格式
        filter_options = "TXT Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"

        # 打开保存文件对话框，用户选择文件类型
        file_path, selected_filter = QFileDialog.getSaveFileName(None, "Save result", global_var.folder_path,
                                                                 filter_options)

        if file_path:
            # 根据选择的文件类型添加适当的扩展名（如果用户没有输入文件扩展名）
            if selected_filter == "CSV Files (*.csv)" and not file_path.endswith('.csv'):
                file_path += ".csv"
            elif selected_filter == "TXT Files (*.txt)" and not file_path.endswith('.txt'):
                file_path += ".txt"
            elif selected_filter == "Excel Files (*.xlsx)" and not file_path.endswith('.xlsx'):
                file_path += ".xlsx"

            # 根据文件类型保存
            if selected_filter == "Excel Files (*.xlsx)":
                # 使用pandas保存为Excel文件
                text.to_excel(file_path, index=False)  # 不保存索引列
                print(f"Excel file saved to {file_path}")

            elif selected_filter == "CSV Files (*.csv)":
                # 使用pandas保存为CSV文件
                text.to_csv(file_path, index=False, encoding='utf-8-sig')  # 不保存索引列，制定编码方式
                print(f"CSV file saved to {file_path}")

            else:  # 保存为TXT
                # 将DataFrame转换为CSV格式的文本，然后写入TXT文件
                text_str = text.to_csv(index=False, sep=' ')
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text_str)  # 保存为TXT文件
                print(f"Text file saved to {file_path}")

def draw_scatter(ax, name, num, target, finalData, data=None):
    """
            绘制不同维度的散点图。

            参数:
            ax: Matplotlib 的 Axes 对象
            name: 主成分名称列表
            num: 维度数量（2 或 3）
            target: 目标变量（类别标签）
            finalData: PCA 转换后的数据
            data: 原始数据（仅在 num==0 时使用）
            """
    print(finalData)
    if num == 3:
        # 创建一个颜色映射对象
        cmap = cm.get_cmap('tab10')

        # 获取类别值的唯一值和对应的索引
        unique_categories = np.unique(target)

        # 绘制所有数据点，并为每个类别使用不同的颜色
        for i, category in enumerate(unique_categories):
            category_data = finalData[target == category]  # 提取出 finalData 中对应当前类别的数据行
            color = cmap(i % cmap.N)
            ax.scatter(category_data[:, 0], category_data[:, 1], category_data[:, 2], color=color,
                                 label=f"Category {category}")
            # print(category_data[:, 0].shape, category_data[:, 1].shape, category_data[:, 2].shape)
        # 添加图例，只显示每种颜色的标签
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = list(set(handles))
        unique_labels = [label for handle, label in zip(handles, labels) if handle in unique_handles]
        ax.legend(unique_handles, unique_labels, title="Category", loc='upper right', bbox_to_anchor=(1.4, 1))
        # 设置坐标轴标签
        if len(name) != 0:
            ax.set_xlabel("PC" + name[0])
            ax.set_ylabel("PC" + name[1])
            ax.set_zlabel("PC" + name[2])
        else:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

    elif num == 2:
        # 创建一个颜色映射对象
        cmap = cm.get_cmap('tab10')

        # 获取类别值的唯一值和对应的索引
        unique_categories = np.unique(target)

        # 绘制所有数据点，并为每个类别使用不同的颜色
        for i, category in enumerate(unique_categories):
            category_data = finalData[target == category]
            color = cmap(i % cmap.N)
            ax.scatter(category_data[:, 0], category_data[:, 1], color=color, label=f'{category}')

        # 添加图例，只显示每种颜色的标签
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = list(set(handles))
        unique_labels = [label for handle, label in zip(handles, labels) if handle in unique_handles]
        ax.legend(unique_handles, unique_labels, title="Category", loc='upper right')

        # 设置坐标轴标签
        if len(name) != 0:
            ax.set_xlabel("PC" + name[0])
            ax.set_ylabel("PC" + name[1])
        else:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

    elif num == 0:
        cols = data.columns[1:]  # 假设第一列是目标变量
        for column in cols:
            ax.scatter(data[column], target, label=column)

        # 添加标题和标签
        ax.set_title('Scatter Plot of Specific Columns with Target Variable')
        ax.set_xlabel('Independent Variables')
        ax.set_ylabel('Dependent Variable')

        # 添加图例
        ax.legend()

class PCASHOW(QDialog, Ui_Dimension_show):  # 继承 QDialog
    def __init__(self, parent=None):
        super(PCASHOW, self).__init__(parent)  # 设置父窗口
        self.setupUi(self)  # 设置 UI 界面
        type_set(self)
        self.ButInit()
        self.Dinum = self.Di_spinBox.value()
        self.Contrnum = self.Di_spinBox_2.value()
        self.file_path = ' '

    def ButInit(self):
        # 绑定按钮点击事件
        self.pushButton.clicked.connect(self.num_select)
        self.toolButton.clicked.connect(self.select_file)


    def num_select(self):
        """弹出文件夹选择对话框"""
        self.Dinum = self.Di_spinBox.value()
        self.Contrnum = self.Di_spinBox_2.value()
        self.accept()  # 关闭对话框并返回 QDialog.Accepted

    def select_file(self):
        """弹出文件夹选择对话框"""
        folder_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)")
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            self.FilePath_lineEdit.setText(folder_path)  # 显示到 QLineEdit
            self.file_path = folder_path

    def plot_scree_plot(self, ax, variance_ratios):
        # fig, ax = plt.subplots()  # 正确解包
        """绘制 PCA 碎石图"""
        ax.plot(range(1, len(variance_ratios) + 1), variance_ratios, 'o-', label='单个主成分贡献率')
        ax.plot(range(1, len(variance_ratios) + 1), np.cumsum(variance_ratios), 's-', label='累积贡献率')
        ax.axhline(y=0.8, color='r', linestyle='--', label='80% 阈值')  # 标记目标阈值
        ax.set_xlabel("主成分数量")
        ax.set_ylabel("方差贡献率")
        ax.set_title("PCA 碎石图")
        ax.legend()
        ax.grid()


class LDASHOW(QDialog, Ui_Dimension_show):  # 继承 QDialog
    def __init__(self, parent=None):
        super(LDASHOW, self).__init__(parent)  # 设置父窗口
        self.setupUi(self)  # 设置 UI 界面
        type_set(self)
        self.label.setText("LDA维度（2-传感器数）")
        self.ButInit()
        self.Dinum = self.Di_spinBox.value()
        self.Contrnum = self.Di_spinBox_2.value()

    def ButInit(self):
        # 绑定按钮点击事件
        self.pushButton.clicked.connect(self.num_select)
        self.toolButton.clicked.connect(self.select_file)

    def num_select(self):
        """弹出文件夹选择对话框"""
        self.Dinum = self.Di_spinBox.value()
        self.Contrnum = self.Di_spinBox_2.value()
        self.accept()  # 关闭对话框并返回 QDialog.Accepted

    def select_file(self):
        """弹出文件夹选择对话框"""
        folder_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)")
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            self.FilePath_lineEdit.setText(folder_path)  # 显示到 QLineEdit
            self.file_path = folder_path

    def plot_lda_variance_ratio(self, ax, variance_ratios, Contrnum):
        """绘制 LDA 解释方差比图"""
        # 绘制单个主成分的解释方差比
        # fig, ax = plt.subplots()  # 正确解包
        ax.plot(range(1, len(variance_ratios) + 1), variance_ratios, 'o-', label='单个主成分解释方差比')

        # 绘制累积解释方差比
        cumulative_variance_ratios = np.cumsum(variance_ratios)
        ax.plot(range(1, len(cumulative_variance_ratios) + 1), cumulative_variance_ratios, 's-', label='累积解释方差比')

        # 标记目标阈值（例如 80%）
        ax.axhline(y=0.8, color='r', linestyle='--', label=f"{Contrnum}% 阈值")

        # 设置坐标轴标签和标题
        ax.set_xlabel("主成分数量")
        ax.set_ylabel("解释方差比")
        ax.set_title("LDA 解释方差比图")

        # 添加图例
        ax.legend()

        # 添加网格
        ax.grid()


class SVMSHOW(QDialog, Ui_SVM_UI):
    def __init__(self, parent=None):
        super(SVMSHOW, self).__init__(parent)  # 设置父窗口
        self.setupUi(self)  # 设置 UI 界面
        type_set(self)
        self.comboBox.addItems(["线性核", "多项式核", "RBF核", "Sigmoid核",
                                    "预计算核"])
        self.kernel_map = {
            "线性核": "linear",
            "多项式核": "poly",
            "RBF核": "rbf",
            "Sigmoid核": "sigmoid",
            "预计算核": "precomputed"
        }
        self.ButInit()

    def ButInit(self):
        # 绑定按钮点击事件
        self.pushButton_2.clicked.connect(self.num_select)
        self.toolButton.clicked.connect(self.select_file)

    def num_select(self):
        """选择了选择模型：关闭对话框"""
        self.desize = self.doubleSpinBox.value()
        self.linear = self.kernel_map[self.comboBox.currentText()]
        self.C = self.doubleSpinBox_C.value()
        self.accept()  # 关闭对话框并返回 QDialog.Accepted

    def select_file(self):
        """弹出文件夹选择对话框"""
        folder_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)")
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            self.FilePath_lineEdit.setText(folder_path)  # 显示到 QLineEdit
            self.file_path = folder_path


class KNNSHOW(QDialog, Ui_KNN_UI):
    def __init__(self, parent=None):
        super(KNNSHOW, self).__init__(parent)  # 设置父窗口
        self.setupUi(self)  # 设置 UI 界面
        type_set(self)
        self.comboBox.addItems(["自动", "Ball Tree 算法", "KD Tree 算法", "暴力计算"])
        self.knn_map = {
            "自动": "auto",
            "Ball Tree 算法": "ball_tree",
            "KD Tree 算法": "kd_tree",
            "暴力计算": "brute"
        }
        self.ButInit()

    def ButInit(self):
        # 绑定按钮点击事件
        self.pushButton_2.clicked.connect(self.num_select)
        self.toolButton.clicked.connect(self.select_file)

    def num_select(self):
        """选择了选择模型：关闭对话框"""
        self.desize = self.doubleSpinBox.value()
        self.alg = self.knn_map[self.comboBox.currentText()]
        self.N = self.N_spinBox.value()
        self.accept()  # 关闭对话框并返回 QDialog.Accepted

    def select_file(self):
        """弹出文件夹选择对话框"""
        folder_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)")
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            self.FilePath_lineEdit.setText(folder_path)  # 显示到 QLineEdit
            self.file_path = folder_path


class LRSHOW(QDialog, Regression_uishow):  # 继承 QDialog
    def __init__(self, parent=None):
        super(LRSHOW, self).__init__(parent)  # 设置父窗口
        self.setupUi(self)  # 设置 UI 界面
        self.label.setText("测试集占比(0.2)")
        self.ButInit()
        type_set(self)
        self.desize = self.doubleSpinBox.value()


    def ButInit(self):
        # 绑定按钮点击事件
        self.pushButton_2.clicked.connect(self.num_select)
        self.toolButton.clicked.connect(self.select_file)

    def num_select(self):
        """选择了选择模型：关闭对话框"""
        self.desize = self.doubleSpinBox.value()
        self.accept()  # 关闭对话框并返回 QDialog.Accepted

    def select_file(self):
        """弹出文件夹选择对话框"""
        folder_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)")
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            self.FilePath_lineEdit.setText(folder_path)  # 显示到 QLineEdit
            self.file_path = folder_path




class PRESHOW(QDialog, Ui_Pre_show):  # 继承 QDialog
    def __init__(self, parent=None):
        super(PRESHOW, self).__init__(parent)  # 设置父窗口
        self.setupUi(self)  # 设置 UI 界面
        type_set(self)
        self.ButInit()
        self.Pre_ComboBox.addItems(["不选", "算术平均滤波法", "递推平均滤波法", "中位值平均滤波法",
                           "一阶滞后滤波法", "加权递推平均滤波法", "消抖滤波法", "限幅消抖滤波法"])
        self.Valchoose_ComboBox.addItems(["不选", "平均值", "中位数", "众数","极差", "最大值","最大斜率"])
        self.PreAlg = self.Pre_ComboBox.currentText()
        self.ValAlg = self.Valchoose_ComboBox.currentText()
        self.file_path = ' '

    def ButInit(self):
        # 绑定按钮点击事件
        self.pushButton.clicked.connect(self.Alg_select) # 完成设置
        self.toolButton.clicked.connect(self.select_file) # 选择文件

    def Alg_select(self):
        """弹出文件夹选择对话框"""
        self.PreAlg = self.Pre_ComboBox.currentText()
        self.ValAlg = self.Valchoose_ComboBox.currentText()
        self.accept()  # 关闭对话框并返回 QDialog.Accepted

    def select_file(self):
        """弹出文件夹选择对话框"""
        folder_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)")
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            self.FilePath_lineEdit.setText(folder_path)  # 显示到 QLineEdit
            self.file_path = folder_path

    def plot_data(self, ax, data, result):
        data = data.iloc[1:, 1:]
        data = data.apply(pd.to_numeric, errors='coerce')
        data.plot(ax=ax)  # 在传入的 ax 上绘图
        ax.set_title("原始数据")

    def plot_result(self, ax, data, result):
        data = result.iloc[1:, 1:]
        data = data.apply(pd.to_numeric, errors='coerce')
        # ax = data.plot()  # 直接使用 DataFrame 的 plot 方法,按列绘制
        data.plot(ax=ax)  # 在传入的 ax 上绘图
        ax.set_title("滤波后数据")


def plot_confusion(ax, confusion , labels_name, is_norm=True, colorbar = True):
    import numpy as np
    import matplotlib.pyplot as plt
    if is_norm == True:
        confusion = np.around(confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis], 2)  # 横轴归一化并保留2位小数

    """绘制混淆矩阵"""
    confusion = np.array(confusion)  # 将list类型转换成数组类型，如果已经是numpy数组类型，则忽略此步骤。
    # 显示归一化后的混淆矩阵
    im = ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(len(confusion)):
        for j in range(len(confusion)):
            ax.annotate(confusion[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')  # 默认所有值均为黑色
    if colorbar:
        plt.colorbar(im, ax=ax)  # 创建颜色条
    # 获取标签的间隔数
    num_class = np.arange(len(labels_name))

    # 设置坐标轴
    ax.set_xticks(num_class)
    ax.set_xticklabels(labels_name, rotation=90)
    ax.set_yticks(num_class)
    ax.set_yticklabels(labels_name)

    # 设置坐标轴标签
    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')

# ? 绘制决策边界图 函数
def plot_decision_regions(ax, X, y, classifier, resolution=0.02):
    print(X, y)

    # 初始化 LabelEncoder
    label_encoder = LabelEncoder()

    # 对训练标签进行编码
    y_train_encoded = label_encoder.fit_transform(y)

    # 设置标记生成器和颜色图
    markers = ('s', '^', 'o', 'x', 'v', 'D', 'p', 'H')  # 更多标记生成器
    colors = plt.cm.get_cmap('Set1', len(np.unique(y_train_encoded)))  # 动态设置颜色

    cmap = ListedColormap(colors(np.arange(len(np.unique(y_train_encoded)))))

    # 绘制决策曲面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z_encoded = label_encoder.transform(Z)  # 使用编码器转换
    Z_encoded = Z_encoded.reshape(xx1.shape)

    # 绘制决策边界
    ax.contourf(xx1, xx2, Z_encoded, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # 绘制所有样本散点图
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0],
                   y=X[y == cl, 1],
                   alpha=0.8,
                   c=colors(idx),  # 使用动态颜色
                   marker=markers[idx],
                   label=cl,
                   edgecolor='black')

    # 添加图例
    ax.legend(loc='upper left')


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
