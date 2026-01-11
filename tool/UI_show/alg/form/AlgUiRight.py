# -*- coding: utf-8 -*-
import json
import os

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                               QPushButton, QHBoxLayout, QTextEdit,
                               QFrame, QMessageBox, QApplication, QListWidgetItem, QSpinBox, QDoubleSpinBox, QCheckBox,
                               QComboBox, QLineEdit)

from tool.UI_show.alg import AlgModelWorker, AlgModelParameters
from tool.UI_show.alg.AlgPreprocessWorker import PreprocessWorker
import pandas as pd
from datetime import datetime

from tool.UI_show.alg.Zoom import ZoomableLabel


class RightFrameManager:
    def __init__(self, ui_instance):
        """初始化右侧Tab管理器"""
        self.ui = ui_instance

        # 初始化变量
        self.tab_counters = {  # 用于跟踪Tab页数量
            "碎石图": 0,
            "散点图": 0,
            "原数据": 0
        }
        self.open_tabs = {}  # 存储打开的Tab页

        # 计算结果
        self.calculate_result = None

        # 连接信号与槽
        self.setup_connections()

        # 初始化预处理相关变量
        self.preprocess_thread = None
        self.preprocess_worker = None

    def setup_connections(self):
        """设置信号与槽的连接"""
        # 连接Tab切换事件
        self.ui.tabWidget.currentChanged.connect(self.on_tab_changed)
        # 连接数据合并文件类型变化事件
        self.ui.comboBox_file_type.currentTextChanged.connect(self.on_file_type_changed)
        # 识别CSV文件
        self.ui.btn_identify_csv.clicked.connect(self.on_identify_csv)
        # 数据预处理槽函数
        self.ui.btn_preprocess.clicked.connect(self.on_btn_preprocess_clicked)
        # 数据合并
        self.ui.btn_merge.clicked.connect(self.on_merge)

        # 连接模型类型变化事件
        self.ui.comboBox_model_type.currentTextChanged.connect(self.on_model_type_changed)
        # 具体模型的变化事件
        self.ui.comboBox_model.currentTextChanged.connect(self.on_model_changed)
        # 计算
        self.ui.btn_calculate.clicked.connect(self.on_btn_calculate_clicked)
        # 显示图片按钮
        self.ui.btn_show_image.clicked.connect(self.on_btn_show_image_clicked)
        # 特征重要性另存为
        self.ui.btn_save_feature_importance.clicked.connect(self.on_btn_save_feature_importance_clicked)
        # 模型信息另存为
        self.ui.btn_save_model_info.clicked.connect(self.on_btn_save_model_info_clicked)
        # 用于存储动态创建的参数控件和标签
        self.param_widgets = {}  # 存储参数控件，键为参数名
        self.param_labels = {}  # 存储参数标签，键为参数名

    def on_model_type_changed(self, text):
        """模型类型选择变化事件"""
        if text == "分类模型":
            self.ui.comboBox_model.clear()
            self.ui.comboBox_model.addItems(["--请选择--", "LDA", "SVM", "RF"])
            self.ui.comboBox_model.setEnabled(True)
            self.ui.comboBox_model.setCurrentIndex(0)
        elif text == "预测模型":
            self.ui.comboBox_model.clear()
            self.ui.comboBox_model.addItems(["--请选择--", "KNN", "BP-ANN", "PLSR", "MLR", "SVR"])
            self.ui.comboBox_model.setEnabled(True)
            self.ui.comboBox_model.setCurrentIndex(0)
        elif text == "降维模型":
            self.ui.comboBox_model.clear()
            self.ui.comboBox_model.addItems(["--请选择--", "PCA", "LDA", "LLE"])
            self.ui.comboBox_model.setEnabled(True)
            self.ui.comboBox_model.setCurrentIndex(0)
        else:
            self.ui.comboBox_model.clear()
            self.ui.comboBox_model.addItems(["请先选择模型类型"])
            self.ui.comboBox_model.setEnabled(False)

    def on_model_changed(self, text):
        """当具体模型改变时，动态生成参数设置表单"""
        model_type = self.ui.comboBox_model_type.currentText()
        model_name = text

        # 清除之前的参数设置
        self.clear_param_form()

        # 如果没有选择模型，直接返回
        if model_name == "请先选择模型类型" or model_name == "":
            return

        # 根据模型类型和具体模型生成参数
        parameters = AlgModelParameters.get_param(model_type, model_name)
        if parameters:
            self.create_param_form(parameters)

        # 设置图片默认参数
        image_init_param = json.dumps(AlgModelParameters.get_image_param(model_type, model_name), ensure_ascii=False, indent=2)
        self.ui.textEdit_image_params.setText(image_init_param)

    def create_param_form(self, parameters):
        """根据参数定义创建表单"""
        self.ui.label_params.setVisible(False)
        row = 0
        for param in parameters:
            label_txt = param["label"]
            if label_txt == "无" or label_txt == "":
                label_txt = ""
            label = QLabel(label_txt)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            label.setFixedWidth(150)
            label.setToolTip(param["tip"])

            widget = self.create_param_widget(param)
            # 存储控件引用
            self.param_widgets[param["name"]] = widget
            self.param_labels[param["name"]] = param["label"]

            self.ui.gridLayout_params.addWidget(label, row, 0)
            self.ui.gridLayout_params.addWidget(widget, row, 1)
            row += 1

    def create_param_widget(self, param):
        """根据参数类型创建对应的输入控件"""
        param_type = param.get("type", "str")
        default = param.get("default", "")

        if param_type == "int":
            widget = QSpinBox()
            widget.setMinimum(param.get("min", -999999))
            widget.setMaximum(param.get("max", 999999))
            widget.setValue(int(default) if default and default != "" else 0)
        elif param_type == "float":
            widget = QDoubleSpinBox()
            widget.setMinimum(param.get("min", -999999.0))
            widget.setMaximum(param.get("max", 999999.0))
            widget.setDecimals(param.get("decimals", 2))
            widget.setValue(float(default) if default and default != "" else 0.0)
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

        # 设置最小宽度，使控件显示完整
        if not isinstance(widget, QTextEdit):
            widget.setMinimumWidth(200)

        widget.setToolTip(param["tip"])
        return widget

    def clear_param_form(self):
        """清除参数设置区域的所有动态控件"""
        self.ui.label_params.setVisible(False)
        for i in reversed(range(self.ui.gridLayout_params.count())):
            widget = self.ui.gridLayout_params.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        self.param_labels.clear()
        self.param_widgets.clear()

    def on_btn_preprocess_clicked(self):
        """数据预处理按钮点击事件"""
        try:
            # 1. 获取选中的文件
            selected_files = self.ui.left_manager.get_selected_files()
            if not selected_files:
                QMessageBox.warning(self.ui, "警告", "请先在左侧文件树中选择要处理的文件")
                return

            # 2. 获取数据计算目录
            data_dir = self.ui.left_manager.get_data_directory_path()
            if not data_dir or not os.path.exists(data_dir):
                QMessageBox.warning(self.ui, "警告", "请先导入文件夹并设置数据计算目录")
                return

            # 3. 获取文件类型
            file_type = self.ui.comboBox_file_type.currentText()

            # 4. 验证文件类型与选中文件的一致性
            if file_type == "TXT":
                # 检查是否都是TXT文件
                for file in selected_files:
                    if not file.lower().endswith('.txt'):
                        QMessageBox.warning(self.ui, "警告",
                                            f"文件 '{os.path.basename(file)}' 不是TXT文件\n请确保所有选中文件都是TXT格式")
                        return

                target_col = None
                data_columns = []

            else:  # CSV
                # 检查是否都是CSV文件
                for file in selected_files:
                    if not file.lower().endswith('.csv'):
                        QMessageBox.warning(self.ui, "警告",
                                            f"文件 '{os.path.basename(file)}' 不是CSV文件\n请确保所有选中文件都是CSV格式")
                        return

                # 获取target列
                target_col = self.ui.comboBox_target.currentText()
                if target_col == "--请选择--":
                    target_col = None

                # 获取选中的数据列
                data_columns = []
                for i in range(self.ui.listWidget_data_columns.count()):
                    item = self.ui.listWidget_data_columns.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        data_columns.append(item.text())

                if not data_columns:
                    QMessageBox.warning(self.ui, "警告", "请至少选择一个数据列")
                    return

            # 5. 获取预处理函数选择
            filter_func = self.ui.comboBox_filter.currentText()
            value_func = self.ui.comboBox_value.currentText()

            # 6. 验证至少选择了一种处理方法
            if filter_func == "--请选择--" and value_func == "--请选择--":
                QMessageBox.warning(self.ui, "警告", "请至少选择一种处理方法（滤波函数或值选择）")
                return

            # 7. 确认处理
            reply = QMessageBox.question(
                self.ui, "确认处理",
                f"即将处理 {len(selected_files)} 个文件\n文件类型: {file_type}\n"
                f"滤波函数: {filter_func}\n值选择: {value_func}\n"
                f"确定开始处理吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # 8. 禁用预处理按钮，防止重复点击
            self.ui.btn_preprocess.setEnabled(False)
            self.ui.btn_preprocess.setText("处理中...")

            # 9. 在工作线程中进行预处理
            self.start_preprocessing(selected_files, data_dir, file_type,
                                     target_col, data_columns, filter_func, value_func)

        except Exception as e:
            QMessageBox.critical(self.ui, "错误", f"预处理设置错误: {str(e)}")

    def start_preprocessing(self, files, data_dir, file_type, target_col,
                            data_columns, filter_func, value_func):
        """启动预处理线程"""
        # 创建工作线程
        self.preprocess_worker = PreprocessWorker(
            files, data_dir, file_type, target_col, data_columns,
            filter_func, value_func, self.ui
        )

        # 创建工作线程
        self.preprocess_thread = QThread()
        self.preprocess_worker.moveToThread(self.preprocess_thread)

        # 连接信号
        self.preprocess_worker.progress.connect(self.on_preprocess_progress)
        self.preprocess_worker.finished.connect(self.on_preprocess_finished)
        self.preprocess_worker.file_processed.connect(self.on_file_processed)

        self.preprocess_thread.started.connect(self.preprocess_worker.run)

        # 线程结束时清理
        self.preprocess_worker.finished.connect(self.preprocess_thread.quit)
        self.preprocess_worker.finished.connect(self.preprocess_worker.deleteLater)
        self.preprocess_thread.finished.connect(self.preprocess_thread.deleteLater)

        # 启动线程
        self.preprocess_thread.start()

    def on_preprocess_progress(self, progress, message):
        """更新预处理进度"""
        self.ui.bottom_manager.update_status(message)

    def on_file_processed(self, filename, status):
        """单个文件处理完成"""
        # 可以在UI中显示处理状态
        current_text = self.ui.textEdit_results.toPlainText()
        new_text = f"{filename}: {status}\n"
        self.ui.textEdit_results.append(new_text)

    def on_preprocess_finished(self, success, message):
        """预处理完成"""
        # 启用预处理按钮
        self.ui.btn_preprocess.setEnabled(True)
        self.ui.btn_preprocess.setText("数据预处理")

        # 显示完成消息
        if success:
            QMessageBox.information(self.ui, "完成", message)
            # 刷新数据计算目录
            #self.refresh_data_directory()
            data_dir = self.ui.left_manager.get_data_directory_path()
            if data_dir:
                # 重新扫描目录并更新文件树
                self.ui.left_manager.refresh_directory(data_dir)
        else:
            QMessageBox.warning(self.ui, "完成", message)

    def closeEvent(self, event):
        """窗口关闭时停止处理线程"""
        if self.preprocess_worker:
            self.preprocess_worker.stop()
            if self.preprocess_thread and self.preprocess_thread.isRunning():
                self.preprocess_thread.quit()
                self.preprocess_thread.wait(2000)  # 等待2秒
        event.accept()

        # 清空工作线程引用
        self.preprocess_thread = None
        self.preprocess_worker = None

    def on_file_type_changed(self, file_type):
        """文件类型选择变化事件"""
        if file_type == "CSV":
            # 启用指定target、数据列和识别CSV文件按钮
            self.ui.comboBox_target.setEnabled(True)
            self.ui.listWidget_data_columns.setEnabled(True)
            self.ui.btn_identify_csv.setEnabled(True)
            self.ui.bottom_manager.update_status("已选择CSV文件类型，请点击'识别CSV文件'按钮")
        elif file_type == "TXT":
            # 禁用指定target、数据列和识别CSV文件按钮
            self.ui.comboBox_target.setEnabled(False)
            self.ui.listWidget_data_columns.setEnabled(False)
            self.ui.btn_identify_csv.setEnabled(False)

            # 清空相关控件
            self.ui.comboBox_target.clear()
            self.ui.comboBox_target.addItem("--请选择--")
            self.ui.listWidget_data_columns.clear()

            self.ui.bottom_manager.update_status("已选择TXT文件类型")
        else:
            # 默认禁用
            self.ui.comboBox_target.setEnabled(False)
            self.ui.listWidget_data_columns.setEnabled(False)
            self.ui.btn_identify_csv.setEnabled(False)
            self.ui.bottom_manager.update_status("请选择文件类型")

    def on_identify_csv(self):
        """识别CSV文件按钮点击事件"""
        selected_files = self.ui.left_manager.get_selected_files()

        if not selected_files:
            QMessageBox.warning(self.ui, "警告", "请先在左侧选择要处理的CSV文件！")
            return

        csv_files = [f for f in selected_files if f.lower().endswith('.csv')]

        if not csv_files:
            QMessageBox.warning(self.ui, "警告", "选中的文件中没有CSV文件！")
            return

        try:
            # 读取第一个CSV文件的列信息
            first_file = csv_files[0]
            import pandas as pd
            df = pd.read_csv(first_file)

            column_names = df.columns.tolist()

            consistent = True
            for csv_file in csv_files[1:]:
                try:
                    other_df = pd.read_csv(csv_file)
                    if other_df.columns.tolist() != column_names:
                        consistent = False
                        QMessageBox.warning(self.ui, "警告",
                                            f"文件结构不一致！\n{first_file} 与 {csv_file} 的列结构不同。")
                        self.ui.comboBox_target.clear()
                        self.ui.comboBox_target.addItem("--请选择--")
                        self.ui.listWidget_data_columns.clear()
                        break
                except Exception as e:
                    QMessageBox.warning(self.ui, "警告", f"读取文件 {csv_file} 时出错: {str(e)}")
                    return

            if consistent:
                # 更新指定target下拉框
                self.ui.comboBox_target.clear()
                self.ui.comboBox_target.addItem("--请选择--")
                self.ui.comboBox_target.addItems(column_names)

                # 更新数据列多选列表
                self.ui.listWidget_data_columns.clear()
                for col in column_names:
                    item = QListWidgetItem(col)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked)  # 默认全选
                    self.ui.listWidget_data_columns.addItem(item)

                # 更新状态栏和结果文本框
                file_count = len(csv_files)
                column_count = len(column_names)

                self.ui.bottom_manager.update_status(f"已识别 {file_count} 个CSV文件，共 {column_count} 列")

                # 在结果文本框中显示识别结果
                csv_files_info = "\n".join([f"  - {os.path.basename(f)}" for f in csv_files])

                self.ui.textEdit_results.setPlainText(
                    f"CSV文件识别结果:\n"
                    f"已识别文件数: {file_count}\n"
                    f"文件列表:\n{csv_files_info}\n\n"
                    f"列信息（共 {column_count} 列）:\n"
                    f"{', '.join(column_names)}\n\n"
                    f"请在'指定target'下拉框中选择目标列，"
                    f"并在'数据列'中选择要合并的列。"
                )

                # 打印到控制台
                print("=" * 50)
                print(f"识别到 {file_count} 个CSV文件:")
                for i, file_path in enumerate(csv_files, 1):
                    print(f"  文件{i}: {os.path.basename(file_path)} (路径: {file_path})")
                print(f"列结构: {column_names}")
                print("=" * 50)

        except Exception as e:
            QMessageBox.critical(None, "错误", f"识别CSV文件时出错:\n{str(e)}")
            self.ui.bottom_manager.update_status("CSV文件识别失败")

    def on_merge(self):
        """数据合并按钮点击事件"""
        try:
            selected_files = self.ui.left_manager.get_selected_files()
            if not selected_files or len(selected_files) == 0:
                QMessageBox.warning(self.ui, "警告", "请先选择要合并的文件！")
                return

            file_type = self.ui.comboBox_file_type.currentText()
            if file_type == "--请选择--" or not file_type:
                QMessageBox.warning(self.ui, "警告", "请在文件类型设置中选择文件类型！")
                return

            for file_path in selected_files:
                ext = os.path.splitext(file_path)[1].lower()
                if file_type == "TXT" and ext != '.txt':
                    QMessageBox.warning(self.ui, "警告",
                                        f"选择的文件类型为TXT，但文件 '{os.path.basename(file_path)}' 不是TXT格式！")
                    return
                elif file_type == "CSV" and ext != '.csv':
                    QMessageBox.warning(self.ui, "警告",
                                        f"选择的文件类型为CSV，但文件 '{os.path.basename(file_path)}' 不是CSV格式！")
                    return

            data_dir = self.ui.left_manager.get_data_directory_path()
            if not data_dir or not os.path.exists(data_dir):
                QMessageBox.warning(self.ui, "警告", "数据计算目录不存在！")
                return

            if file_type == "TXT":
                success, result_file = self._merge_txt_files(selected_files, data_dir)
            else:  # CSV
                success, result_file = self._merge_csv_files(selected_files, data_dir)

            if success:
                self.ui.left_manager.refresh_directory(data_dir)

                QMessageBox.information(self.ui, "成功",
                                        f"文件合并成功！\n\n已保存到：\n{result_file}\n\n"
                                        f"共合并了 {len(selected_files)} 个文件。")
            else:
                QMessageBox.warning(self.ui, "警告", "文件合并失败，请检查文件格式和参数设置！")

        except Exception as e:
            QMessageBox.critical(self.ui, "错误", f"合并文件时出错：\n{str(e)}")

    def _merge_txt_files(self, file_paths, data_dir):
        """合并TXT文件"""
        try:
            all_data = []
            data_columns_count = None
            file_names = []

            for file_path in file_paths:
                # 读取TXT文件
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='gbk') as f:
                            lines = f.readlines()
                    except:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            lines = f.readlines()

                # 解析数据
                file_data = []
                for line in lines:
                    line = line.strip()
                    if line:  # 跳过空行
                        # 分割数据（支持空格、制表符等空白字符）
                        parts = line.split()
                        # 转换为浮点数
                        try:
                            row_data = [float(part) for part in parts]
                            file_data.append(row_data)
                        except ValueError:
                            raise ValueError(f"文件 {os.path.basename(file_path)} 包含非数值数据：{line}")

                if not file_data:
                    raise ValueError(f"文件 {os.path.basename(file_path)} 为空或格式不正确")

                # 检查数据维度是否一致
                row_lengths = [len(row) for row in file_data]
                if len(set(row_lengths)) > 1:
                    raise ValueError(f"文件 {os.path.basename(file_path)} 数据行长度不一致：{row_lengths}")

                current_columns = len(file_data[0])

                # 记录第一个文件的列数作为标准
                if data_columns_count is None:
                    data_columns_count = current_columns
                elif data_columns_count != current_columns:
                    raise ValueError(
                        f"文件 {os.path.basename(file_path)} 的列数({current_columns})与其他文件不一致({data_columns_count})")

                # 获取文件名（不含扩展名）作为target
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                # 为每行数据添加target（文件名）
                for row in file_data:
                    new_row = [file_name] + row
                    all_data.append(new_row)

                file_names.append(file_name)

            if not all_data:
                raise ValueError("没有有效数据可以合并")

            # 创建列名：第一列为target，后面为sensor1, sensor2, ...
            column_names = ['target']
            for i in range(data_columns_count):
                column_names.append(f'sensor{i + 1}')

            # 创建DataFrame
            df = pd.DataFrame(all_data, columns=column_names)

            # 8. 生成新文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_count = len(file_paths)
            new_filename = f"merged_txt_{file_count}files_{timestamp}.csv"
            new_filepath = os.path.join(data_dir, new_filename)

            # 保存为CSV文件
            df.to_csv(new_filepath, index=False, encoding='utf-8')

            return True, new_filepath

        except Exception as e:
            raise Exception(f"合并TXT文件时出错：{str(e)}")

    def _merge_csv_files(self, file_paths, data_dir):
        """合并CSV文件"""
        try:
            target_col = self.ui.comboBox_target.currentText()
            if target_col == "--请选择--" or not target_col:
                raise ValueError("请选择target列！")

            data_columns = []
            for i in range(self.ui.listWidget_data_columns.count()):
                item = self.ui.listWidget_data_columns.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    data_columns.append(item.text())

            if not data_columns:
                raise ValueError("请至少选择一个数据列！")

            if target_col in data_columns:
                raise ValueError(f"数据列中不能包含target列 '{target_col}'")

            all_data_frames = []

            for file_path in file_paths:
                try:
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(file_path, encoding='gbk')
                        except:
                            df = pd.read_csv(file_path, encoding='latin-1')
                except Exception as e:
                    raise ValueError(f"读取文件 {os.path.basename(file_path)} 失败：{str(e)}")

                if df.empty:
                    raise ValueError(f"文件 {os.path.basename(file_path)} 为空")

                required_columns = [target_col] + data_columns
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"文件 {os.path.basename(file_path)} 缺少以下列：{', '.join(missing_columns)}")

                non_numeric_columns = []
                for col in data_columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                    except:
                        # 如果转换失败，检查是否所有值都可以转换为数值
                        non_numeric_values = []
                        for idx, value in enumerate(df[col]):
                            try:
                                pd.to_numeric([value], errors='raise')
                            except:
                                non_numeric_values.append(f"第{idx + 1}行: {str(value)}")

                        if non_numeric_values:
                            non_numeric_columns.append(f"{col} (非数值数据: {', '.join(non_numeric_values[:3])}" +
                                                       ("..." if len(non_numeric_values) > 3 else ""))

                if non_numeric_columns:
                    error_msg = f"文件 {os.path.basename(file_path)} 的以下列包含非数值数据：\n"
                    error_msg += "\n".join(non_numeric_columns[:5])  # 只显示前5个错误
                    if len(non_numeric_columns) > 5:
                        error_msg += f"\n...等{len(non_numeric_columns)}个列"
                    raise ValueError(error_msg)

                df_selected = df[required_columns].copy()

                if target_col != 'target':
                    df_selected = df_selected.rename(columns={target_col: 'target'})
                    column_order = ['target'] + data_columns
                    df_selected = df_selected[column_order]

                all_data_frames.append(df_selected)

            if not all_data_frames:
                raise ValueError("没有有效数据可以合并")

            merged_df = pd.concat(all_data_frames, ignore_index=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_count = len(file_paths)
            data_count = len(data_columns)

            safe_target_name = target_col.replace(' ', '_').replace('/', '_').replace('\\', '_')
            new_filename = f"merged_csv_{safe_target_name}_{data_count}cols_{file_count}files_{timestamp}.csv"
            new_filepath = os.path.join(data_dir, new_filename)

            # 保存为CSV文件
            merged_df.to_csv(new_filepath, index=False, encoding='utf-8')

            return True, new_filepath

        except Exception as e:
            raise Exception(f"合并CSV文件时出错：{str(e)}")

    def on_btn_calculate_clicked(self):
        """点击计算按钮时，读取动态创建的控件值"""
        # 检查是否有动态创建的参数控件
        if not self.param_widgets:
            self.ui.textEdit_results.setText("错误：请先选择模型并设置参数！")
            return

        model_type = self.ui.comboBox_model_type.currentText()
        model_name = self.ui.comboBox_model.currentText()
        image_size = self.ui.comboBox_image_size.currentText()
        # 图片参数，必须是一个合法的json格式
        image_param = self.ui.textEdit_image_params.toPlainText()
        image_param_json = {}
        try:
            if image_param != "":
                image_param_json = json.loads(image_param)
        except Exception as e:
            QMessageBox.warning(self.ui, "警告", "图篇参数设置错误，必须是一个合法的json格式")
            return

        dpi = 100
        if image_size == "小":
            dpi = 60
        elif image_size == "较小":
            dpi = 80
        elif image_size == "较大":
            dpi = 140
        elif image_size == "大":
            dpi = 200

        if model_type == "--请选择--" or model_name == "请先选择模型类型":
            self.ui.textEdit_results.setText("错误：请先选择模型类型和具体模型！")
            return

        # 读取所有参数值
        params = {}
        for param_name, widget in self.param_widgets.items():
            param_label = self.param_labels.get(param_name, param_name)

            if isinstance(widget, QSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[param_name] = widget.currentText()
            elif isinstance(widget, QTextEdit):
                params[param_name] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                params[param_name] = widget.text()
            elif isinstance(widget, QLabel):
                params[param_name] = widget.text()

        # 生成结果文本
        result_text = f"【模型信息】\n"
        result_text += f"模型类型: {model_type}\n"
        result_text += f"具体模型: {model_name}\n"
        result_text += f"生成图片大小: {self.ui.comboBox_image_size.currentText()}\n\n"

        result_text += "【参数设置】\n"
        if params:
            for param_name, param_value in params.items():
                param_label = self.param_labels.get(param_name, param_name)
                result_text += f"{param_label}: {param_value}\n"
        else:
            result_text += "无参数设置\n"

        result_text += "\n【计算结果】\n"
        result_text += "点击'计算'按钮已读取所有参数。\n"
        result_text += "实际计算功能需结合具体算法实现。"

        # 显示结果
        self.ui.textEdit_results.setText(result_text)

        # 更新状态栏
        self.ui.bottom_manager.update_status(f"已读取{len(params)}个参数，模型类型: {model_type}, 具体模型: {model_name}")

        # 打印参数到控制台
        print(f"模型类型: {model_type}")
        print(f"具体模型: {model_name}")
        print(f"参数值: {params}")
        work_path = self.ui.left_manager.get_data_directory_path()
        selected_files = self.ui.left_manager.get_selected_files()
        if len(selected_files) == 0:
            QMessageBox.warning(self.ui, "警告", "请选择计算的数据文件")
            return
        if len(selected_files) > 1:
            QMessageBox.warning(self.ui, "警告", "请选择一个计算的数据文件")
            return
        selected_file = selected_files[0]
        ext = os.path.splitext(selected_file)[1].lower()
        if ext != ".csv":
            QMessageBox.warning(self.ui, "警告", "请选择CSV格式的计算的数据文件")
            return

        # 点击计算前先设置结果中的控件信息禁用
        self.ui.comboBox_image_selector.setEnabled(False)
        self.ui.comboBox_image_selector.setStyleSheet("background-color: grey;")
        self.ui.btn_show_image.setEnabled(False)
        self.ui.btn_show_image.setStyleSheet("background-color: grey;")
        self.ui.btn_save_feature_importance.setEnabled(False)
        self.ui.btn_save_feature_importance.setStyleSheet("background-color: grey;")
        self.ui.btn_save_model_info.setEnabled(False)
        self.ui.btn_save_model_info.setStyleSheet("background-color: grey;")
        self.ui.comboBox_image_selector.clear()
        self.ui.comboBox_image_selector.addItem("--请选择--")

        in_param = {}
        in_param["params"] = params
        in_param["image_param"] = image_param_json

        self.worker = AlgModelWorker.ModelWorker(work_path, selected_file, model_type, model_name, in_param, dpi)
        self.ui.tabWidget.setCurrentIndex(2)
        self.worker.finished.connect(self.on_calculate_work_finished)
        self.worker.error.connect(self.on_calculate_work_error)
        self.worker.start()

    def on_calculate_work_finished(self, result):
        print("结果展示-------\n")
        print(result)
        result_json = json.loads(result)
        self.calculate_result = result_json
        self.ui.textEdit_results.setText(result_json["summary"])  # 显示算法结果

        if result_json["success"]:
            self.ui.comboBox_image_selector.setEnabled(True)
            self.ui.comboBox_image_selector.setStyleSheet("background-color: #3ba4a7;")
            self.ui.btn_show_image.setEnabled(True)
            self.ui.btn_show_image.setStyleSheet("background-color: #3ba4a7;")
            self.ui.btn_save_feature_importance.setEnabled(True)
            self.ui.btn_save_feature_importance.setStyleSheet("background-color: #3ba4a7;")
            self.ui.btn_save_model_info.setEnabled(True)
            self.ui.btn_save_model_info.setStyleSheet("background-color: #3ba4a7;")

        # 读取返回的imgs

        imgs = result_json["generated_files"]["imgs"]
        _img_names = []
        for img in imgs:
            _img_names.append(img["name"])
            if img["name"] not in self.tab_counters:
                self.tab_counters[img["name"]] = 0

        self.ui.comboBox_image_selector.addItems(_img_names)

        # self.ui.left_manager.refresh_directory(self.ui.left_manager.get_data_directory_path())

    def on_calculate_work_error(self, error_msg):
        self.ui.textEdit_results.setText(error_msg)
        self.ui.comboBox_image_selector.setEnabled(False)
        self.ui.comboBox_image_selector.setStyleSheet("background-color: grey;")
        self.ui.btn_show_image.setEnabled(False)
        self.ui.btn_show_image.setStyleSheet("background-color: grey;")
        self.ui.btn_save_feature_importance.setEnabled(False)
        self.ui.btn_save_feature_importance.setStyleSheet("background-color: grey;")
        self.ui.btn_save_model_info.setEnabled(False)
        self.ui.btn_save_model_info.setStyleSheet("background-color: grey;")

    def close_tab(self, tab_id):
        """关闭指定的Tab页"""
        if tab_id in self.open_tabs:
            tab_widget = self.open_tabs[tab_id]

            index = self.ui.tabWidget.indexOf(tab_widget)
            if index >= 0:
                self.ui.tabWidget.removeTab(index)
                del self.open_tabs[tab_id]
                self.ui.bottom_manager.update_status(f"已关闭Tab页: {tab_id}")

    def on_tab_changed(self, index):
        if index >= 0:
            tab_text = self.ui.tabWidget.tabText(index)
            self.ui.bottom_manager.update_status(f"当前: {tab_text}")

    def on_btn_show_image_clicked(self):
        """ 显示图片的槽函数 """
        if not self.calculate_result:
            QMessageBox.warning(self.ui, "警告", "请先进行计算")
            return
        select_img = self.ui.comboBox_image_selector.currentText()
        if select_img == "--请选择--":
            QMessageBox.warning(self.ui, "警告", "请选择要显示的图片")
            return

        all_imgs = self.calculate_result["generated_files"]["imgs"]

        for img in all_imgs:
            if img["name"] == select_img:
                self.open_plot_tab(img["name"], img["img"], img["data"])

    def on_btn_save_feature_importance_clicked(self):
        if not self.calculate_result:
            QMessageBox.warning(self.ui, "警告", "请先进行计算")
            return
        feature_importance_path = self.calculate_result["generated_files"]["feature_importance"]["data"]

        self.save_plot_data("特征重要性", feature_importance_path)

    def on_btn_save_model_info_clicked(self):
        if not self.calculate_result:
            QMessageBox.warning(self.ui, "警告", "请先进行计算")
            return

        model_info_path = self.calculate_result["generated_files"]["model_info"]["data"]

        self.save_plot_data("模型信息", model_info_path)


    def open_plot_tab(self, plot_type, img_path, data_path):
        """打开图片显示Tab页"""
        # 检查是否已经打开了太多Tab页
        if self.ui.tabWidget.count() >= 10:  # 限制最多10个Tab页
            QMessageBox.warning(self.ui, "警告", "打开的Tab页过多，请先关闭一些Tab页。")
            return

        # 增加计数器
        self.tab_counters[plot_type] += 1
        tab_id = f"{plot_type}_{self.tab_counters[plot_type]}"

        # 创建新的Tab页
        new_tab = QWidget()
        new_tab.setObjectName(tab_id)
        new_tab.plot_type = plot_type
        new_tab.img_path = img_path
        new_tab.data_path = data_path

        # 创建布局
        layout = QVBoxLayout(new_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 创建内容框架 - 直接显示图片，去掉title部分
        content_frame = QFrame()
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # 创建图片显示区域
        image_label = ZoomableLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("""
            QLabel {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)

        # 加载图片
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            image_label.setText(f"无法加载图片: {img_path}")
        else:
            image_label.setPixmap(pixmap)
            # 设置初始大小为图片原始大小，但不超过Tab区域
            max_width = self.ui.tabWidget.width() - 50
            max_height = self.ui.tabWidget.height() - 150

            if pixmap.width() > max_width or pixmap.height() > max_height:
                scaled_pixmap = pixmap.scaled(
                    max_width, max_height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
                image_label.original_pixmap = scaled_pixmap

        content_layout.addWidget(image_label, 1)  # 1表示占用所有可用空间

        # 创建底部按钮框架
        button_frame = QFrame()
        button_frame.setFrameShape(QFrame.Shape.StyledPanel)
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setSpacing(10)  # 设置按钮间距

        # 创建底部按钮 - 去掉颜色和大小控制，让按钮自动填充
        save_image_btn = QPushButton("图片另存为")
        save_image_btn.clicked.connect(lambda: self.save_plot_image(image_label, plot_type, img_path))

        save_data_btn = QPushButton("图片数据另存为")
        save_data_btn.clicked.connect(lambda: self.save_plot_data(plot_type, data_path))

        reset_zoom_btn = QPushButton("重置缩放")
        reset_zoom_btn.clicked.connect(image_label.reset_zoom)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(lambda: self.close_tab(tab_id))

        button_layout.addStretch()
        button_layout.addWidget(save_image_btn)
        button_layout.addWidget(save_data_btn)
        button_layout.addWidget(reset_zoom_btn)
        button_layout.addWidget(close_btn)
        button_layout.addStretch()

        # 将各个部分添加到主布局
        layout.addWidget(content_frame, 1)  # 1表示图片区域占用所有可用空间
        layout.addWidget(button_frame)

        # 添加Tab页到TabWidget
        self.ui.tabWidget.addTab(new_tab, plot_type)

        # 切换到新创建的Tab页
        self.ui.tabWidget.setCurrentWidget(new_tab)

        # 存储Tab页引用
        self.open_tabs[tab_id] = new_tab

        # 更新状态栏
        self.ui.bottom_manager.update_status(f"已打开{plot_type}")

    def save_plot_image(self, image_label, plot_type, default_path):
        """保存图片"""
        success, message = image_label.save_image(default_path)
        if success:
            QMessageBox.information(self.ui, "成功", message)
        else:
            if message != "取消保存":
                QMessageBox.warning(self.ui, "警告", message)

    def save_plot_data(self, plot_type, data_path):
        """保存图片数据"""
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self.ui, "警告", f"{plot_type}数据文件不存在")
            return

        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self.ui, f"保存{plot_type}数据",
            os.path.basename(data_path),
            "CSV文件 (*.csv);;TXT文件 (*.txt);;所有文件 (*.*)"
        )

        if file_path:
            try:
                import shutil
                shutil.copy2(data_path, file_path)
                QMessageBox.information(self.ui, "成功", f"数据已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self.ui, "错误", f"保存数据失败: {str(e)}")