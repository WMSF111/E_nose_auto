
import os
import tempfile
import shutil
from datetime import datetime
from PySide6.QtWidgets import *
from PySide6.QtCore import QThread, Signal, Slot, Qt
import traceback
import pandas as pd

from resource_ui.alg_puifile.ui_Alg_Frame import Ui_Form
import data_file.model_LDA
import data_file.model_SVM
import data_file.model_RF
import data_file.model_ANN
import data_file.model_KNN
import data_file.model_PLSR
import data_file.model_MLR
import data_file.model_SVR
import data_file.model_PCA
import data_file.model_LDA2
import data_file.model_LLE
from tool.UI_show.Alg_show_scv import CSVTableWidget
from tool.UI_show.Alg_show_png import PngViewer

import data_file.filter as filter


# 滤波函数
FilterFuns = ["--请选择--", "算术平均滤波法", "递推平均滤波法", "中位值平均滤波法", "一阶滞后滤波法",
              "加权递推平均滤波法", "消抖滤波法", "限幅消抖滤波法"]
# 值选择
ValFuns = ["--请选择--", "平均值", "中位数", "众数", "极差", "最大值", "最大斜率"]

Model_Class = ["--请选择--", "分类模型", "预测模型", "降维模型"]       # 算法分类
Model_Classify = ["LDA", "SVM", "RF"]                                # 分类模型 算法
Model_Prediction = ["KNN", "BP-ANN", "PLSR", "MLR", "SVR"]           # 预测模型 算法
Model_Dimension = ["PCA", "LDA", "LLE"]                              # 降维模型 算法

def calculate_slope(group):
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

def get_file_name(file_path):
    """从文件路径中获得文件名"""
    return os.path.basename(file_path)

class AlgUiFrame(QWidget, Ui_Form):
    def __init__(self):
        super(AlgUiFrame, self).__init__()
        self.setupUi(self)  # 设置 UI 界面

        self.setStyleSheet("""
            QPlainTextEdit{
                background-color: white;
                color: black;
            }
        """)

        # 工作路径
        self.work_item = QTreeWidgetItem(self.treeWidget, 0)  # 0目录  1文件
        self.work_item.setText(0, "[work path]")
        self.work_path = os.path.join(tempfile.gettempdir(), 'enose')
        print("工作路径: ", self.work_path)
        if not os.path.exists(self.work_path):
            try:
                os.makedirs(self.work_path)  # 创建目录
            except OSError as e:
                print(f"创建目录失败: {self.work_path}, 错误: {e}")
        self.sub_files(self.work_item, self.work_path)  # 获得工作路径下的文件
        self.work_item.setExpanded(True)  # 展开


        self.data_items = []            # 导入的数据目录
        self.source_file_path = ""      # 原数据文件路径

        # tree
        self.treeWidget.setHeaderHidden(True)  # 隐藏表头
        # self.treeWidget.setFrameShape(QFrame::NoFrame)  # 设置无边框
        # self.treeWidget.setFocusPolicy(Qt::NoFocus)  # 去除选中虚线框

        # 初始化 滤波函数 值选择 comboBox
        self.cBox_FilterFun.addItems(FilterFuns)
        self.cBox_ValSelect.addItems(ValFuns)

        # 算法按钮使能
        self.btn_scree_plot.setEnabled(False)
        self.btn_scree_plot.setStyleSheet("background-color: grey;")
        self.btn_scatter_plot.setEnabled(False)
        self.btn_scatter_plot.setStyleSheet("background-color: grey;")

        # 初始化 算法分类 算法 comboBox
        self.cBox_AlgGroup.addItems(Model_Class)

        # 算法参数
        self.input_widgets = {}
        parameters = [{"name": "init", "label": "无", "type": "str", "default": ""}]
        self.create_form(parameters)  # 在input_widgets中添加此widget

        # 算法进度条
        self.progressBar.setVisible(False)

        # 按钮
        self.btn_import.clicked.connect(self.import_dir)                    # 导入数据目录
        self.btn_clear_work_path.clicked.connect(self.clear_work_dir)       # 清空工作目录
        self.btn_unchecked_all.clicked.connect(self.unchecked_all)          # 取消所有勾选

        self.btn_Merge.clicked.connect(self.files_merge_txt)                # 合并文件
        self.btn_Pre.clicked.connect(self.files_pre)                        # 预处理文件
        self.btn_clear_pre.clicked.connect(self.clear_pre_out)              # 清空预处理结果输出

        self.cBox_AlgGroup.currentTextChanged.connect(self.on_alg_group_changed)    # 算法分类comboBox选择事件
        self.cBox_AlgName.currentTextChanged.connect(self.on_alg_name_changed)      # 算法名称comboBox选择事件
        self.cBox_AlgGroup.model().item(0).setEnabled(False)                        # 禁止选择"--请选择--"
        self.btn_run.clicked.connect(self.on_btn_run_click)                         # 计算 按钮事件
        self.btn_scree_plot.clicked.connect(self.show_scree_plot)                   # 显示碎石图 按钮事件
        self.btn_scatter_plot.clicked.connect(self.show_scatter_plot)               # 显示散点图 按钮事件
        self.btn_source_data.clicked.connect(self.show_source_data)                 # 显示元数据 按钮事件

        # tab关闭
        self.tabWidget.setTabsClosable(True)    # 启用所有关闭按钮
        self.tabWidget.tabCloseRequested.connect(self.on_tab_close_requested)       # tab关闭事件

        # splitter 左右占比 3:7
        self.splitter.setSizes([3000, 7000])

    def on_tab_close_requested(self, index):
        """关闭标签页"""
        # 前三个tab不能关闭
        if self.tabWidget.tabText(index) == "使用说明" or self.tabWidget.tabText(index) == "数据预处理"\
                or self.tabWidget.tabText(index) == "模型计算":
            return

        widget = self.tabWidget.widget(index)
        if widget:
            # 从tab widget中移除
            self.tabWidget.removeTab(index)
            # 安排删除（在事件循环中删除）
            widget.deleteLater()
        else:
            self.tabWidget.removeTab(index)

    # ------------------ 左侧tree ------------------------------

    def import_dir(self):
        """弹出文件夹选择对话框"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择数据文件夹",  # 对话框标题
            "",  # 默认打开的目录（空表示当前目录）
        )
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            # 添加数据目录
            path_item = QTreeWidgetItem(self.treeWidget, 0)
            path_item.setText(0, folder_path)
            self.data_items.append(path_item)  # 添加到数据目录集合
            self.sub_files(path_item, folder_path)  # 获得下级所有文件
            # 将工作目录移动到最下面
            for i in range(self.treeWidget.topLevelItemCount()):
                item = self.treeWidget.topLevelItem(i)
                if item.text(0) == "[work path]":
                    current_index = self.treeWidget.indexOfTopLevelItem(item)
                    removed_item = self.treeWidget.takeTopLevelItem(current_index)  # 移除
                    self.treeWidget.addTopLevelItem(removed_item)  # 重新插入到底部

    def sub_files(self, parent_item, parent_path):
        """获得下级所有文件，添加到parent_item节点下"""
        # 目录下的文件
        all_items = os.listdir(parent_path)  # 列出目录下所有内容（包括文件和文件夹）
        # print("目录下的文件all_items: ", all_items)
        files = [item for item in all_items if os.path.isfile(os.path.join(parent_path, item))]  # 筛选出文件
        # print("目录下的文件: ", files)
        for file in files:
            file_item = QTreeWidgetItem(parent_item, 1)
            file_item.setText(0, os.path.basename(file))
            file_item.setCheckState(0, Qt.Unchecked)
        parent_item.setExpanded(True)  # 展开

    def unchecked_all(self):
        """取消所有勾选"""
        iterator = QTreeWidgetItemIterator(self.treeWidget)
        while iterator.value():
            item = iterator.value()
            if item.type() == 1:  # 文件
                item.setCheckState(0, Qt.CheckState.Unchecked)
            iterator += 1

    def clear_work_dir(self):
        """清空工作路径"""
        items = os.listdir(self.work_path)  # 获取目录内容
        if not items:
            return  # 目录已经是空的
        # 删除所有文件和目录
        for item in items:
            item_path = os.path.join(self.work_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)        # 删除文件
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)    # 上传目录
        # 清空节点下的内容
        while self.work_item.childCount() > 0:
            child = self.work_item.takeChild(0)
            del child  # 删除子节点对象
        # 获得工作路径下的文件
        self.sub_files(self.work_item, self.work_path)
        self.work_item.setExpanded(True)  # 展开

    def flush_work_path(self):
        """刷新工作目录"""
        # 清空节点下的内容
        while self.work_item.childCount() > 0:
            child = self.work_item.takeChild(0)
            del child  # 删除子节点对象
        self.sub_files(self.work_item, self.work_path) # 重新获得工作目录下的文件

    def get_checked_files(self):
        """获得所有选择的文件"""
        checked_items = []  # 选择的节点
        iterator = QTreeWidgetItemIterator(self.treeWidget)
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == Qt.CheckState.Checked:
                checked_items.append(item)
            iterator += 1
        files = []  # 选择的文件
        for item in checked_items:
            parent = item.parent()
            if parent.text(0) == "[work path]":
                file_path = os.path.join(self.work_path, item.text(0))
            else:
                file_path = os.path.join(parent.text(0), item.text(0))
            files.append(file_path)
        return files

    # ------------------ 预处理 ------------------------------

    def files_merge_txt(self):
        """txt文件合并"""
        checked_files = self.get_checked_files()  # 勾选的文件名
        print("要合并的文件:", checked_files)
        if len(checked_files) == 0:
            QMessageBox.warning(None, "错误", "请勾选要合并的文件")
            return

        # 合并文件前，先获得第一个文件的类型和列数
        if checked_files[0].lower().endswith(".csv"):
            file_type = "csv"   # csv文件类型
        else:
            file_type = "txt"   # txt文件类型
        with open(checked_files[0], 'r', encoding='utf-8') as file:
            content = file.read()
        if content is None:
            return
        lines = content.splitlines()
        if len(lines) == 0:
            QMessageBox.warning(None, "错误", f"第一个文件空内容: {checked_files[0]}")
            return
        first_len_cols = lines[0].split()
        first_len_cols_num = len(first_len_cols)    # 列数

        # 合并文件
        if file_type == "txt":   # txt文件类型
            out = ""  # 文件内容
            for n in range(first_len_cols_num + 1):  # 第一行（列名）
                if n == 0:
                    out += "target,"
                else:
                    if n == first_len_cols_num:
                        out += f"sensor{n}"
                    else:
                        out += f"sensor{n},"
            out += "\n"
            for file_name in checked_files:  # 内容（遍历所有选择文件）
                with open(file_name, 'r', encoding='utf-8') as file:
                    content = file.read()
                lines = content.splitlines()  # 分割行
                # print("文件行: ", lines)
                for line in lines:
                    line = line.strip()
                    if not line:  # 空行
                        continue
                    cols = line.split()  # 分割列
                    # print("列: ", cols)
                    if len(cols) != first_len_cols_num:
                        QMessageBox.warning(None, "错误", f"文件列数不一致: {file_name}")
                        return
                    row = [get_file_name(file_name)] + cols  # 第一列是原文件名
                    out_line = ",".join(row)
                    print("行: ->" + out_line + "<-")
                    out += out_line + "\n"
            # 写文件 data.csv
            out_file_path = os.path.join(self.work_path, "data.csv")
            try:
                with open(out_file_path, 'w', encoding='utf-8') as file:
                    file.write(out)
                timestamp = int(datetime.now().timestamp())  # 整数秒时间戳
                dt = datetime.fromtimestamp(timestamp)
                out = f"==== {dt} ====\n"
                out += "文件已合并到 [work_path]/data.csv"
                self.pTextEdit_pre.appendPlainText(out)  # 输出
            except Exception as e:
                QMessageBox.warning(None, "错误", "文件合并失败")
            self.flush_work_path()  # 刷新工作目录
        else:       # csv文件类型
            dfs = []
            for file_name in checked_files:  # 遍历所有选择文件
                df = pd.read_csv(file_name)
                dfs.append(df)  # 先保存到数组中
            combined_df = pd.concat(dfs, ignore_index=True, sort=False)  # 合并df
            out_file_path = os.path.join(self.work_path, "data.csv")
            combined_df.to_csv(out_file_path, index=False)  # 保存文件
            timestamp = int(datetime.now().timestamp())  # 整数秒时间戳
            dt = datetime.fromtimestamp(timestamp)
            out = f"==== {dt} ====\n"
            out += "文件已合并到 [work_path]/data.csv"
            self.pTextEdit_pre.appendPlainText(out)  # 输出
            self.flush_work_path()  # 刷新工作目录

    def files_pre(self):
        """数据文件预处理"""
        checked_files = self.get_checked_files()  # 勾选的文件名
        print("要预处理的文件:", checked_files)
        if len(checked_files) == 0:
            QMessageBox.warning(None, "错误", "请勾选要预处理的文件")
            return

        for file_path in checked_files:
            filter_funname = self.cBox_FilterFun.currentText()
            val_funname = self.cBox_ValSelect.currentText()
            if filter_funname == "--请选择--" and val_funname == "--请选择--":
                QMessageBox.warning(None, "错误", "请选择滤波函数和值选择")
                return
            # 数据集
            df = pd.read_csv(file_path, header=0)  # 读文件到DataFrame，第0行作为列名
            print(df)
            df_data = df.iloc[:, 1:].copy()  # 去除target列
            df_result = df.copy()  # 结果数据集
            # 滤波函数处理
            if filter_funname != "--请选择--":
                for column in df_data.columns:
                    column_data = df_data[column].to_numpy()  # 转化为数组
                    if filter_funname == FilterFuns[1]:  # 算术平均滤波法
                        # window_size: 窗口大小，用于计算中位值，输入整数，越小越接近原数据
                        result = filter.ArithmeticAverage(column_data.copy(), 2)
                    elif filter_funname == FilterFuns[2]:  # 递推平均滤波法
                        result = filter.SlidingAverage(column_data.copy(), 2)
                    elif filter_funname == FilterFuns[3]:  # 中位值平均滤波法
                        result = filter.MedianAverage(column_data.copy(), 2)
                    elif filter_funname == FilterFuns[4]:  # 一阶滞后滤波法
                        # 滞后程度决定因子，0~1（越大越接近原数据）
                        result = filter.FirstOrderLag(column_data.copy(), 0.9)
                    elif filter_funname == FilterFuns[5]:  # 加权递推平均滤波法
                        # 平滑系数，范围在0到1之间（越大越接近原数据）
                        result = filter.WeightBackstepAverage(column_data.copy(), 0.9)
                    elif filter_funname == FilterFuns[6]:  # 消抖滤波法
                        # N:消抖上限,范围在2以上。
                        result = filter.ShakeOff(column_data.copy(), 4)
                    elif filter_funname == FilterFuns[7]:  # 限幅消抖滤波法
                        # Amplitude:限制最大振幅,范围在0 ~ ∞ 建议设大一点
                        # N:消抖上限,范围在0 ~ ∞
                        result = filter.AmplitudeLimitingShakeOff(column_data.copy(), 200, 3)
                    # 将 result 列表中的元素显式转换为目标列的数据类型
                    result = [df_result[column].dtype.type(value) for value in result]
                    # 将处理后的数据直接更新回原始的 DataFrame
                    df_result.iloc[:, df_result.columns.get_loc(column)] = result
            # 值选择处理
            if val_funname != "--请选择--":
                df_result.set_index(df_result.columns[0], inplace=True)  # 第一列(target)作为索引
                df_result = df_result.apply(pd.to_numeric, errors='coerce')  # 全部转化为数字
                if val_funname == ValFuns[1]:  # 平均值
                    df_new = df_result.groupby(df_result.index).median()  # 按平均值新建dataframe
                elif val_funname == ValFuns[2]:  # 中位数
                    df_new = df_result.groupby(df_result.index).mean()
                elif val_funname == ValFuns[3]:  # 众数
                    # mode()，它会返回一个 DataFrame 或者 Series，包含每个组的众数（如果有多个众数，它会列出所有的众数）
                    # iloc[0] 用来选取第一个众数（如果有多个众数）
                    df_new = (df_result.groupby(df_result.index).agg(lambda x: x.mode().iloc[0]))
                elif val_funname == ValFuns[4]:  # 极差
                    df_new = df_result.groupby(df_result.index).agg(lambda x: x.max() - x.min())
                elif val_funname == ValFuns[5]:  # 最大值
                    df_new = df_result.groupby(df_result.index).agg(lambda x: x.max())
                elif val_funname == ValFuns[6]:  # 最大斜率
                    # 计算每组相邻数据点的斜率
                    df_new = df_result.groupby(df_result.index).apply(calculate_slope)
                # 确保返回的 DataFrame 与原始数据类型一致
                df_new = df_new.astype(df_result.iloc[:, 0].dtype)
                # 取消索引，将 'target' 列恢复为普通列
                df_new.reset_index(inplace=True)
                # 更新结果数据集
                df_result = df_new

            # 写文件 pre_data.csv
            out_file_path = os.path.join(self.work_path, f"pre_{get_file_name(file_path)}")
            df_result.to_csv(out_file_path, index=False)
            timestamp = int(datetime.now().timestamp())  # 整数秒时间戳
            dt = datetime.fromtimestamp(timestamp)
            out = f"==== {dt} ====\n"
            out += "预处理文件已输出到 [work_path]"
            self.pTextEdit_pre.appendPlainText(out)  # 输出
            self.flush_work_path()  # 刷新工作目录

    def clear_pre_out(self):
        """清空预处理结果输出"""
        self.pTextEdit_pre.clear()

    # ------------------ 算法 ------------------------------

    def on_alg_group_changed(self, text):
        """算法分类comboBox选择事件"""
        if text == Model_Class[1]:      # 分类模型
            self.cBox_AlgName.clear()
            self.cBox_AlgName.addItems(Model_Classify)
        elif text == Model_Class[2]:    # 预测模型
            self.cBox_AlgName.clear()
            self.cBox_AlgName.addItems(Model_Prediction)
        elif text == Model_Class[3]:  # 降维模型
            self.cBox_AlgName.clear()
            self.cBox_AlgName.addItems(Model_Dimension)
        else:
            self.cBox_AlgName.clear()

    def on_alg_name_changed(self, text):
        """算法名称comboBox选择事件"""
        group_name = self.cBox_AlgGroup.currentText()
        model_name = self.cBox_AlgName.currentText()
        self.clear_form()  # 清空参数表单
        if group_name == Model_Class[1] and model_name == Model_Classify[0]:        # 分类模型 LDA
            parameters = [
                {"name": "LDA01", "label": "solver(求解算法)", "type": "choice", "options": ["svd", "eigen"], "default": "svd"}
            ]
            self.create_form(parameters)
        elif group_name == Model_Class[1] and model_name == Model_Classify[1]:      # 分类模型 SVM
            parameters = [
                {"name": "SVM01", "label": "kernel(核函数)", "type": "choice",
                 "options": ["linear", "rbf", "poly", "sigmoid"],
                 "default": "rbf"},
                {"name": "SVM02", "label": "C(正则化参数)", "type": "choice",
                 "options": ["0.001", "0.01", "0.1", "1.0", "10", "100", "1000"],
                 "default": "1.0"},
                {"name": "SVM03", "label": "gamma(核系数)", "type": "choice",
                 "options": ["0.001", "scale", "auto", "1.0"],
                 "default": "scale"}
            ]
            self.create_form(parameters)
        elif group_name == Model_Class[1] and model_name == Model_Classify[2]:      # 分类模型 RF
            parameters = [{"name": "init", "label": "无", "type": "str", "default": ""}]
            self.create_form(parameters)
        elif group_name == Model_Class[2] and model_name == Model_Prediction[0]:    # 预测模型 KNN
            parameters = [{"name": "init", "label": "无", "type": "str", "default": ""}]
            self.create_form(parameters)
        elif group_name == Model_Class[2] and model_name == Model_Prediction[1]:    # 预测模型 BP-ANN
            parameters = [
                {"name": "ANN01", "label": "activation(激活函数)", "type": "choice",
                 "options": ["identity", "logistic", "tanh", "relu"],
                 "default": "relu"},
                {"name": "ANN02", "label": "solver(优化算法)", "type": "choice",
                 "options": ["lbfgs", "sgd", "adam"],
                 "default": "adam"},
            ]
            self.create_form(parameters)
        elif group_name == Model_Class[2] and model_name == Model_Prediction[2]:    # 预测模型 PLSR
            parameters = [{"name": "init", "label": "无", "type": "str", "default": ""}]
            self.create_form(parameters)
        elif group_name == Model_Class[2] and model_name == Model_Prediction[3]:    # 预测模型 MLR
            parameters = [{"name": "init", "label": "无", "type": "str", "default": ""}]
            self.create_form(parameters)
        elif group_name == Model_Class[2] and model_name == Model_Prediction[4]:    # 预测模型 SVR
            parameters = [
                {"name": "SVR01", "label": "kernel(核选择)", "type": "choice",
                 "options": ["linear", "poly", "rbf", "sigmoid"],
                 "default": "rbf"},
                {"name": "SVR02", "label": "C(正则化)", "type": "choice",
                 "options": ["0.1", "1.0", "10", "100", "1000"],
                 "default": "1.0"}
            ]
            self.create_form(parameters)
        elif group_name == Model_Class[3] and model_name == Model_Dimension[0]:     # 降维模型 PCA
            parameters = [
                {"name": "PCA01", "label": "n_components(主成分数量)", "type": "choice",
                 "options": ["3", "0.95", "None", "mle"],
                 "default": "None"},
                {"name": "PCA02", "label": "whiten(是否白化)", "type": "choice", "options": ["False", "True"],
                 "default": "False"},
                {"name": "PCA03", "label": "svd_solver(SVD求解器)", "type": "choice", "options": ["auto", "full", "arpack", "randomized"],
                 "default": "auto"}
            ]
            self.create_form(parameters)
        elif group_name == Model_Class[3] and model_name == Model_Dimension[1]:  # 降维模型 LDA
            parameters = [
                {"name": "LDA201", "label": "solver(求解算法)", "type": "choice", "options": ["svd", "eigen"],
                 "default": "svd"}
            ]
            self.create_form(parameters)
        elif group_name == Model_Class[3] and model_name == Model_Dimension[2]:  # 降维模型 LLE
            parameters = [{"name": "init", "label": "无", "type": "str", "default": ""}]
            self.create_form(parameters)

    def on_btn_run_click(self):
        """模型计算"""

        # 模型计算只能选择一个csv原文件
        checked_files = self.get_checked_files()  # 勾选的文件名
        print("要预处理的文件:", checked_files)
        if len(checked_files) != 1:
            QMessageBox.warning(None, "错误", "模型计算需要选择一个原文件")
            return
        self.source_file_path = checked_files[0]
        if not self.source_file_path.lower().endswith('.csv'):
            QMessageBox.warning(None, "错误", "请选择csv原文件")
            return

        group_name = self.cBox_AlgGroup.currentText()   # 分类名称
        model_name = self.cBox_AlgName.currentText()    # 算法名称
        params = []     # 算法参数
        if model_name == "":
            QMessageBox.warning(None, "错误", "请选择算法")
            return
        elif model_name == Model_Classify[0]:
            if group_name == Model_Class[3]:                        # 降维模型 LDA
                params = self.get_values(["LDA201"])
            else:                                                   # 分类模型 LDA
                params = self.get_values(["LDA01"])
        elif model_name == Model_Classify[1]:                       # 分类模型 SVM
            params = self.get_values(["SVM01", "SVM02", "SVM03"])
        elif model_name == Model_Classify[2]:                       # 分类模型 RF
            params = []
        elif model_name == Model_Prediction[0]:                     # 预测模型 KNN
            params = []
        elif model_name == Model_Prediction[1]:                     # 预测模型 BP-ANN
            params = self.get_values(["ANN01", "ANN02"])
        elif model_name == Model_Prediction[2]:                     # 预测模型 PLSR
            params = []
        elif model_name == Model_Prediction[3]:                     # 预测模型 MLR
            params = []
        elif model_name == Model_Prediction[4]:                     # 预测模型 SVR
            params = self.get_values(["SVR01", "SVR02"])
        elif model_name == Model_Dimension[0]:                      # 降维模型 PCA
            params = self.get_values(["PCA01", "PCA02", "PCA03"])
        elif model_name == Model_Dimension[2]:                      # 降维模型 LLE
            params = []

        self.progressBar.setVisible(True)           # 显示进度条
        self.pTextEdit_model.setPlainText("")       # 清空结果框

        # 按钮使能
        self.btn_scree_plot.setEnabled(False)
        self.btn_scree_plot.setStyleSheet("background-color: grey;")
        self.btn_scatter_plot.setEnabled(False)
        self.btn_scatter_plot.setStyleSheet("background-color: grey;")

        dpi = 100
        if self.rbtn_smallMin.isChecked():
            dpi = 60
        elif self.rbtn_small.isChecked():
            dpi = 80
        elif self.rbtn_larger.isChecked():
            dpi = 120
        elif self.rbtn_largerMax.isChecked():
            dpi = 140

        print(f"执行参数{params}\n")
        # 创建并启动工作线程
        self.worker = ModelWorker(self.work_path, self.source_file_path, group_name, model_name, params, dpi)
        self.worker.finished.connect(self.on_work_finished)
        self.worker.error.connect(self.on_work_error)
        self.worker.start()

    @Slot(str)
    def on_work_finished(self, result):
        """处理成功完成"""
        self.pTextEdit_model.setPlainText(result)  # 显示算法结果
        # 按钮使能
        self.btn_scree_plot.setEnabled(True)
        self.btn_scree_plot.setStyleSheet("background-color: #3ba4a7;")
        self.btn_scatter_plot.setEnabled(True)
        self.btn_scatter_plot.setStyleSheet("background-color: #3ba4a7;")
        self.progressBar.setVisible(False)  # 隐藏进度条
        self.flush_work_path()  # 刷新工作目录

    @Slot(str)
    def on_work_error(self, error_msg):
        """处理异常完成"""
        self.pTextEdit_model.setPlainText(error_msg)  # 显示算法异常信息
        self.progressBar.setVisible(False)  # 隐藏进度条

    def create_form(self, parameters):
        """根据参数定义创建表单"""
        for param in parameters:
            label = QLabel(param["label"])
            widget = self.create_input_widget(param)
            self.input_widgets[param["name"]] = widget  # 在input_widgets中添加此widget
            self.formLayout.addRow(label, widget)
        # print("self.input_widgets内容: ", self.input_widgets)

    def create_input_widget(self, param):
        """根据参数类型创建对应的输入控件"""
        param_type = param.get("type", "str")
        default = param.get("default", "")

        if param_type == "int":
            widget = QSpinBox()
            widget.setMinimum(param.get("min", -999999))
            widget.setMaximum(param.get("max", 999999))
            widget.setValue(default if default is not None else 0)
        elif param_type == "float":
            widget = QDoubleSpinBox()
            widget.setMinimum(param.get("min", -999999.0))
            widget.setMaximum(param.get("max", 999999.0))
            widget.setDecimals(param.get("decimals", 2))
            widget.setValue(default if default is not None else 0.0)
        elif param_type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(default))
        elif param_type == "text":
            widget = QTextEdit()
            widget.setText(str(default))
            if "rows" in param:
                widget.setMinimumHeight(param["rows"] * 20)
        elif param_type == "choice":
            widget = QComboBox()
            options = param.get("options", [])
            widget.addItems(options)
            if default in options:
                widget.setCurrentText(default)
        elif param_type == "str":
            widget = QLabel(str(default))
        else:  # 默认为字符串输入
            widget = QLineEdit()
            widget.setText(str(default))
        return widget

    def clear_form(self):
        """清空表单"""
        while self.formLayout.count():
            item = self.formLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.input_widgets.clear()

    def get_values(self, param_names):
        """获取所有参数输入值"""
        values = {}
        for name in param_names:
            widget = self.input_widgets[name]

            if isinstance(widget, QLineEdit):
                values[name] = widget.text()
            elif isinstance(widget, QTextEdit):
                values[name] = widget.toPlainText()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()

        return values

    def show_source_data(self):
        """打开 显示原数据 Tab"""
        tab_index = self.tabWidget.addTab(CSVTableWidget(self.source_file_path), "原数据")
        self.tabWidget.setCurrentIndex(tab_index)

    def show_scree_plot(self):
        """打开 碎石图 Tab"""
        png_path = os.path.join(self.work_path, "scree_plot.png")
        print("碎石图路径: ", png_path)
        tab_index = self.tabWidget.addTab(PngViewer(png_path, self.work_path), "碎石图")
        self.tabWidget.setCurrentIndex(tab_index)

    def show_scatter_plot(self):
        """打开 散点图 Tab"""
        png_path = os.path.join(self.work_path, "scatter_plots.png")
        print("散点图路径: ", png_path)
        tab_index = self.tabWidget.addTab(PngViewer(png_path, self.work_path), "散点图")
        self.tabWidget.setCurrentIndex(tab_index)

class ModelWorker(QThread):
    """线程运行算法"""
    # 定义信号
    finished = Signal(str)      # 成功完成，传递结果
    error = Signal(str)         # 发生错误，传递错误信息

    def __init__(self, dir_path, source_file_path, group_name, model_name, params, dpi):
        super().__init__()
        self.dir_path = dir_path            # 工作目录
        self.src_path = source_file_path    # 原文件路径
        self.group_name = group_name        # 模型分类
        self.model_name = model_name        # 模型名称
        self.params = params                # 参数
        self._is_running = True             # 线程是否运行
        self.dpi = dpi                      # 显示dpi

    def run(self):
        """线程执行"""
        try:
            df = pd.read_csv(self.src_path)         # 读取数据文件
            result = ""                             # 算法结果

            if self.model_name == Model_Classify[0]:
                if self.group_name == Model_Class[3]:                                   # 降维模型 LDA
                    result = data_file.model_LDA2.run(df, self.dir_path, self.params, self.dpi)
                else:                                                                   # 分类模型 LDA
                    result = data_file.model_LDA.run(df, self.dir_path, self.params, self.dpi)
            elif self.model_name == Model_Classify[1]:                                  # 分类模型 SVM
                result = data_file.model_SVM.run(df, self.dir_path, self.params, self.dpi)
            elif self.model_name == Model_Classify[2]:                                  # 分类模型 RF
                result = data_file.model_RF.run(df, self.dir_path, [], self.dpi)
            elif self.model_name == Model_Prediction[0]:                                # 预测模型 KNN
                result = data_file.model_KNN.run(df, self.dir_path, [], self.dpi)
            elif self.model_name == Model_Prediction[1]:                                # 预测模型 BP-ANN
                result = data_file.model_ANN.run(df, self.dir_path, self.params, self.dpi)
            elif self.model_name == Model_Prediction[2]:                                # 预测模型 PLSR
                result = data_file.model_PLSR.run(df, self.dir_path, [], self.dpi)
            elif self.model_name == Model_Prediction[3]:                                # 预测模型 MLR
                result = data_file.model_MLR.run(df, self.dir_path, [], self.dpi)
            elif self.model_name == Model_Prediction[4]:                                # 预测模型 SVR
                result = data_file.model_SVR.run(df, self.dir_path, self.params, self.dpi)
            elif self.model_name == Model_Dimension[0]:                                 # 降维模型 PCA
                result = data_file.model_PCA.run(df, self.dir_path, self.params, self.dpi)
            elif self.model_name == Model_Dimension[2]:                                 # 降维模型 LLE
                result = data_file.model_LLE.run(df, self.dir_path, [], self.dpi)

            self.finished.emit(result)  # 任务完成，发送结果
        except Exception as e:
            # 捕获异常并发送错误信息
            error_msg = f"算法执行出错:\n\n {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def stop(self):
        """停止线程"""
        self._is_running = False
        self.quit()
        self.wait()
