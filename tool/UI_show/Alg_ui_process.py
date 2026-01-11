"""算法执行widget"""

import os
from PySide6.QtWidgets import *
from PySide6.QtCore import QThread, Signal, Slot
import traceback
import pandas as pd
import webbrowser

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
from resource_ui.alg_puifile.ui_AlgProcess import Ui_Form
from tool.UI_show.Alg_show_scv import CSVTableWidget
from tool.UI_show.Alg_show_png import PngViewer

Model_Class = ["--请选择--", "分类模型", "预测模型", "降维模型"]       # 算法分类
Model_Classify = ["LDA", "SVM", "RF"]                                # 分类模型 算法
Model_Prediction = ["KNN", "BP-ANN", "PLSR", "MLR", "SVR"]           # 预测模型 算法
Model_Dimension = ["PCA", "LDA", "LLE"]                              # 降维模型 算法

class AlgProcess(QWidget, Ui_Form):
    def __init__(self, dir_path, parent):
        super(AlgProcess, self).__init__()
        self.setupUi(self)  # 设置 UI 界面
        self.dir_path = dir_path    # data.csv文件所在目录
        self.parent = parent        # 上级Widget（Alg_ui_show.py）
        self.worker = None          # 算法执行线程
        self.setStyleSheet("""
            QPlainTextEdit{
                background-color: white;
                color: black;
                border: none;
            }
        """)

        # 初始化 算法分类 算法 comboBox
        self.cBox_AlgGroup.addItems(Model_Class)

        # 算法参数
        self.input_widgets = {}
        parameters = [{"name": "init", "label": "无", "type": "str", "default": ""}]
        self.create_form(parameters)  # 在input_widgets中添加此widget

        # 按钮使能
        self.btn_scree_plot.setEnabled(False)
        self.btn_scree_plot.setStyleSheet("background-color: grey;")
        self.btn_scatter_plot.setEnabled(False)
        self.btn_scatter_plot.setStyleSheet("background-color: grey;")

        # progressBar显示
        self.progressBar.setVisible(False)

        # 事件
        self.cBox_AlgGroup.currentTextChanged.connect(self.on_alg_group_changed)    # 算法分类comboBox选择事件
        self.cBox_AlgName.currentTextChanged.connect(self.on_alg_name_changed)      # 算法名称comboBox选择事件
        self.cBox_AlgGroup.model().item(0).setEnabled(False)                        # 禁止选择"--请选择--"
        self.btn_run.clicked.connect(self.on_btn_run_click)                         # 计算 按钮事件
        self.btn_scree_plot.clicked.connect(self.show_scree_plot)                   # 显示碎石图 按钮事件
        self.btn_scatter_plot.clicked.connect(self.show_scatter_plot)               # 显示散点图 按钮事件
        self.btn_source_data.clicked.connect(self.show_source_data)                 # 显示元数据 按钮事件


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
        """计算"""

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

        self.progressBar.setVisible(True)       # 显示进度条
        self.plainTextEdit.setPlainText("")     # 清空结果框

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

        # 创建并启动工作线程
        self.worker = ModelWorker(self.dir_path, group_name, model_name, params, dpi)
        self.worker.finished.connect(self.on_work_finished)
        self.worker.error.connect(self.on_work_error)
        self.worker.start()

    @Slot(str)
    def on_work_finished(self, result):
        """处理成功完成"""
        self.plainTextEdit.setPlainText(result)  # 显示算法结果
        # 按钮使能
        self.btn_scree_plot.setEnabled(True)
        self.btn_scree_plot.setStyleSheet("background-color: #3ba4a7;")
        self.btn_scatter_plot.setEnabled(True)
        self.btn_scatter_plot.setStyleSheet("background-color: #3ba4a7;")
        self.progressBar.setVisible(False)  # 隐藏进度条

    @Slot(str)
    def on_work_error(self, error_msg):
        """处理异常完成"""
        self.plainTextEdit.setPlainText(error_msg)  # 显示算法异常信息
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
        file_path = os.path.join(self.dir_path, "data.csv")
        tab_index = self.parent.tabWidget.addTab(CSVTableWidget(file_path), "原数据")
        self.parent.tabWidget.setCurrentIndex(tab_index)

    def show_scree_plot(self):
        """打开 碎石图 Tab"""

        # 方式1： 弹出plt.show()
        # data_file.model_LDA.show_scree_plot(self.model)

        # 方式2：Tab显示
        file_path = os.path.join(self.dir_path, "scree_plot.png")
        tab_index = self.parent.tabWidget.addTab(PngViewer(file_path), "碎石图")
        self.parent.tabWidget.setCurrentIndex(tab_index)

        # 方式3：使用系统看图工具
        # file_path = os.path.join(self.dir_path, "scree_plot.png")
        # os.startfile(file_path)

        # 方式4：使用浏览器
        # file_path = os.path.join(self.dir_path, "scree_plot.png")
        # image_url = f"file://{os.path.abspath(file_path)}"
        # webbrowser.open(image_url)


    def show_scatter_plot(self):
        """打开 散点图 Tab"""
        file_path = os.path.join(self.dir_path, "scatter_plots.png")
        tab_index = self.parent.tabWidget.addTab(PngViewer(file_path), "散点图")
        self.parent.tabWidget.setCurrentIndex(tab_index)

class ModelWorker(QThread):
    """线程运行算法"""

    # 定义信号
    finished = Signal(str)      # 成功完成，传递结果
    error = Signal(str)         # 发生错误，传递错误信息

    def __init__(self, dir_path, group_name, model_name, params, dpi):
        super().__init__()
        self.dir_path = dir_path        # 工作目录
        self.group_name = group_name    # 模型分类
        self.model_name = model_name    # 模型名称
        self.params = params            # 参数
        self._is_running = True         # 线程是否运行
        self.dpi = dpi                  # 显示dpi

    def run(self):
        """线程执行"""
        try:
            file_path = os.path.join(self.dir_path, "data.csv")
            df = pd.read_csv(file_path)         # 读取数据文件
            result = ""                         # 算法结果

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
