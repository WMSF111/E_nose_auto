"""算法模块"""

# import sys
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import *

import pandas as pd
import data_file.filter as filter

# from resource_ui.alg_puifile.Alg_show import Ui_Alg_show  # 导入生成的 UI 类
from resource_ui.alg_puifile.ui_AlgModule import Ui_Form
# import data_file.transfo as transfo
# import matplotlib.pyplot as plt
# import global_var as glov
from data_file.alg_tabadd import ALG_TAB_ADD
from tool.UI_show.Alg_ui_process import AlgProcess

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 滤波函数
FilterFuns = ["--请选择--", "算术平均滤波法", "递推平均滤波法", "中位值平均滤波法", "一阶滞后滤波法",
              "加权递推平均滤波法", "消抖滤波法", "限幅消抖滤波法"]
# 值选择
ValFuns = ["--请选择--", "平均值", "中位数", "众数", "极差", "最大值", "最大斜率"]


def read_file(parent_path, file_name):
    """读文件"""
    file_path = os.path.join(parent_path, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        QMessageBox.warning(None, "错误", f"文件不存在: {file_path}")
        return None
    except PermissionError:
        QMessageBox.warning(None, "错误", f"没有权限读取文件: {file_path}")
        return None
    except UnicodeDecodeError:
        QMessageBox.warning(None, "错误", f"文件编码错误: {file_path}")
        return None


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


class AlgInit(QWidget, Ui_Form):
    def __init__(self):
        super(AlgInit, self).__init__()
        self.setupUi(self)  # 设置 UI 界面
        self.tabadd = ALG_TAB_ADD(self)
        self.setStyleSheet("""
            QWidget {
                color: #333; font: 12pt "Segoe UI";
            }
            QPushButton {
                height: 16px; color: white;
            }
            QTableWidget {
                background-color: white;
            }
        """)

        # 初始化文件列表
        self.table_FileList.setColumnWidth(0, 20)  # 格宽度
        self.table_FileList.horizontalHeader().setStretchLastSection(True)  # 设置充满表宽度
        self.table_FileList.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 禁止编辑
        self.table_FileList.verticalHeader().setVisible(False)  # 隐藏列表头
        self.table_FileList.horizontalHeader().setVisible(False)  # 隐藏行表头

        # 初始化 滤波函数 值选择 comboBox
        self.cBox_FilterFun.addItems(FilterFuns)
        self.cBox_ValSelect.addItems(ValFuns)

        # 按钮事件
        self.btn_SelectDir.clicked.connect(self.select_dir)         # 选择目录按钮事件
        self.btn_Merge.clicked.connect(self.files_merge)            # 合并文件按钮事件
        self.btn_Pre.clicked.connect(self.files_pre)                # 预处理按钮事件
        self.btn_AlgSelect.clicked.connect(self.model_process)      # 模型计算按钮事件

        # tab关闭
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.tabCloseRequested.connect(self.on_tab_close_requested)

        # splitter 左右占比 2:8
        self.splitter.setSizes([2000, 8000])

    def select_dir(self):
        """弹出文件夹选择对话框"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",  # 对话框标题
            "",  # 默认打开的目录（空表示当前目录）
        )
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            self.lineEdit_Dir.setText(folder_path)  # 显示到 QLineEdit

            # 从末尾开始移除行
            for row in reversed(range(self.table_FileList.rowCount())):
                self.table_FileList.removeRow(row)

            all_items = os.listdir(folder_path)  # 目录下所有文件
            for item in all_items:  # 遍历
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):  # 是文件
                    row_position = self.table_FileList.rowCount()  # 当前行数
                    self.table_FileList.insertRow(row_position)  # 插入一行
                    checkbox_item = QTableWidgetItem("")
                    if item_path.endswith(".txt"):
                        checkbox_item.setCheckState(Qt.Checked)  # 选中
                    else:
                        checkbox_item.setCheckState(Qt.Unchecked)
                    self.table_FileList.setItem(row_position, 0, checkbox_item)  # 插入checkbox
                    self.table_FileList.setItem(row_position, 1, QTableWidgetItem(item))  # 插入文件名

    def files_merge(self):
        """文件合并"""
        checked_files = []  # 勾选的文件名
        for row in range(self.table_FileList.rowCount()):
            item = self.table_FileList.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                checked_files.append(self.table_FileList.item(row, 1).text())  # 取文件名
        print("要合并的文件:", checked_files)
        print("文件所在目录:", self.lineEdit_Dir.text())
        if len(checked_files) == 0:
            QMessageBox.warning(None, "错误", "请勾选要合并的文件")
            return
        # 合并文件前，先获得第一个文件的列数
        content = read_file(self.lineEdit_Dir.text(), checked_files[0])
        if content is None:
            return
        lines = content.splitlines()
        if len(lines) == 0:
            QMessageBox.warning(None, "错误", f"第一个文件空内容: {checked_files[0]}")
            return
        first_len_cols = lines[0].split()
        # print("第一个文件第一行:", first_len_cols)
        first_len_cols_num = len(first_len_cols)
        # print("第一个列数:", first_len_cols_num)
        # 合并文件
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
            content = read_file(self.lineEdit_Dir.text(), file_name)
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
                row = [file_name] + cols  # 第一列是原文件名
                out_line = ",".join(row)
                print("行: ->" + out_line + "<-")
                out += out_line + "\n"
        # 写文件 data.csv
        out_file_path = os.path.join(self.lineEdit_Dir.text(), "data.csv")
        try:
            with open(out_file_path, 'w', encoding='utf-8') as file:
                file.write(out)
            QMessageBox.information(None, "提示", "文件已合并到data.csv")
        except Exception as e:
            QMessageBox.warning(None, "错误", "文件合并失败")

    def files_pre(self):
        """数据文件预处理"""
        file_path = os.path.join(self.lineEdit_Dir.text(), "data.csv")
        if not os.path.exists(file_path):
            QMessageBox.warning(None, "错误", "data.csv文件不存在，是否没有做数据合并？")
            return
        filter_funname = self.cBox_FilterFun.currentText()
        val_funname = self.cBox_ValSelect.currentText()
        if filter_funname == "--请选择--" and val_funname == "--请选择--":
            QMessageBox.warning(None, "错误", "请选择滤波函数和值选择")
            return
        # 数据集
        df = pd.read_csv(file_path, header=0)  # 读文件到DataFrame，第0行作为列名
        # print(df)
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
        # 文件更新
        df_result.to_csv(file_path, index=False)
        QMessageBox.information(None, "提示", "预处理已更新到文件data.csv")

    def model_process(self):
        """模型计算"""
        file_path = os.path.join(self.lineEdit_Dir.text(), "data.csv")
        if not os.path.exists(file_path):
            QMessageBox.warning(None, "错误", "data.csv文件不存在，是否没有做数据合并？")
            return
        tab_index = self.tabWidget.addTab(AlgProcess(self.lineEdit_Dir.text(), self), "模型计算")
        self.tabWidget.setCurrentIndex(tab_index)

    def on_tab_close_requested(self, index):
        """关闭标签页"""
        widget = self.tabWidget.widget(index)
        if widget:
            # 从tab widget中移除
            self.tabWidget.removeTab(index)
            # 安排删除（在事件循环中删除）
            widget.deleteLater()
        else:
            self.tabWidget.removeTab(index)

