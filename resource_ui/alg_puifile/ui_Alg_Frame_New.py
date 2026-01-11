# -*- coding: utf-8 -*-

from PySide6.QtCore import (QCoreApplication, QMetaObject, QSize, Qt)
from PySide6.QtGui import (QFont)
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QSpacerItem, QSplitter, QTabWidget,
                               QTreeWidget, QVBoxLayout, QWidget,
                               QTextEdit, QFrame, QListWidget, QGridLayout)


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1165, 866)

        self.verticalLayout_9 = QVBoxLayout(Form)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(5, 5, 5, 5)

        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # ===== 左侧面板：文件树 =====
        self.left_widget = QWidget(self.splitter)
        self.left_widget.setObjectName(u"left_widget")
        self.left_widget.setMinimumSize(QSize(350, 0))

        self.verticalLayout_left = QVBoxLayout(self.left_widget)
        self.verticalLayout_left.setObjectName(u"verticalLayout_left")
        self.verticalLayout_left.setContentsMargins(0, 0, 0, 0)

        self.horizontalLayout_top = QHBoxLayout()
        self.horizontalLayout_top.setObjectName(u"horizontalLayout_top")
        self.horizontalLayout_top.setSpacing(10)

        # 导入文件夹按钮
        self.btn_import = QPushButton(self.left_widget)
        self.btn_import.setObjectName(u"btn_import")
        self.btn_import.setFixedSize(120, 35)

        font_btn = QFont()
        font_btn.setPointSize(12)
        self.btn_import.setFont(font_btn)
        self.horizontalLayout_top.addWidget(self.btn_import)

        self.label_import_hint = QLabel(self.left_widget)
        self.label_import_hint.setObjectName(u"label_import_hint")
        self.label_import_hint.setText("提示: 导入文件夹后选择文件和算法")
        self.label_import_hint.setStyleSheet("color: #666666; font-size: 12px; padding: 5px;")
        self.label_import_hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.horizontalLayout_top.addWidget(self.label_import_hint)

        self.horizontalSpacer_top = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_top.addItem(self.horizontalSpacer_top)
        self.verticalLayout_left.addLayout(self.horizontalLayout_top)

        # 树形控件
        self.treeWidget = QTreeWidget(self.left_widget)
        self.treeWidget.setObjectName(u"treeWidget")

        # 设置树形控件的列
        self.treeWidget.setColumnCount(2)
        self.treeWidget.setHeaderLabels(["文件夹/文件", "选择"])

        # 设置列宽
        self.treeWidget.setColumnWidth(0, 280)
        self.treeWidget.setColumnWidth(1, 60)

        self.treeWidget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        font_tree = QFont()
        font_tree.setPointSize(14)  # 增加字体大小 设置之后方框的大小没有变化
        self.treeWidget.setFont(font_tree)

        self.treeWidget.setStyleSheet("""
            QTreeWidget {
                font-size: 14px;
                outline: 0;  /* 移除焦点边框 */
            }
            QTreeWidget::item {
                height: 25px;  /* 增加行高 */
            }
            /* 隐藏横向滚动条 */
            QTreeWidget QScrollBar:horizontal {
                height: 0px;
            }
        """)

        self.treeWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.treeWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 禁用水平滚动条
        self.verticalLayout_left.addWidget(self.treeWidget)
        self.splitter.addWidget(self.left_widget)

        # ===== 右侧面板 =====
        self.right_widget = QWidget(self.splitter)
        self.right_widget.setObjectName(u"right_widget")
        self.verticalLayout_right = QVBoxLayout(self.right_widget)
        self.verticalLayout_right.setObjectName(u"verticalLayout_right")
        self.verticalLayout_right.setContentsMargins(0, 0, 0, 0)

        self.tabWidget = QTabWidget(self.right_widget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setTabsClosable(False)

        # ===== Tab: 数据预处理 =====
        self.tab_process = QWidget()
        self.tab_process.setObjectName(u"tab_process")
        self.tabWidget.addTab(self.tab_process, "")

        self.verticalLayout_tab2 = QVBoxLayout(self.tab_process)
        self.verticalLayout_tab2.setObjectName(u"verticalLayout_tab2")
        self.verticalLayout_tab2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_tab2.setSpacing(15)  # 增加组件间距

        # ===== 第一个位置：文件选择区域 =====
        self.groupBox_file_select = QGroupBox(self.tab_process)
        self.groupBox_file_select.setObjectName(u"groupBox_file_select")
        self.groupBox_file_select.setTitle("文件类型设置")

        # 设置文件选择区域的大小策略 - 可扩展高度
        self.groupBox_file_select.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # 文件选择布局
        self.formLayout_file_select = QFormLayout(self.groupBox_file_select)
        self.formLayout_file_select.setObjectName(u"formLayout_file_select")
        self.formLayout_file_select.setContentsMargins(15, 15, 15, 15)
        self.formLayout_file_select.setSpacing(15)
        self.formLayout_file_select.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.formLayout_file_select.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # 文件类型选择
        self.label_file_type = QLabel(self.groupBox_file_select)
        self.label_file_type.setObjectName(u"label_file_type")
        self.label_file_type.setText("文件类型:")

        self.comboBox_file_type = QComboBox(self.groupBox_file_select)
        self.comboBox_file_type.setObjectName(u"comboBox_file_type")
        self.comboBox_file_type.addItems(["TXT", "CSV"])
        # 设置下拉框占满可用宽度
        self.comboBox_file_type.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.formLayout_file_select.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_file_type)
        self.formLayout_file_select.setWidget(0, QFormLayout.ItemRole.FieldRole, self.comboBox_file_type)

        # 指定target
        self.label_target = QLabel(self.groupBox_file_select)
        self.label_target.setObjectName(u"label_target")
        self.label_target.setText("指定target:")

        self.comboBox_target = QComboBox(self.groupBox_file_select)
        self.comboBox_target.setObjectName(u"comboBox_target")
        self.comboBox_target.addItems(["--请选择--"])
        self.comboBox_target.setEnabled(False)  # 页面初始化时禁用
        # 设置下拉框占满可用宽度
        self.comboBox_target.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.formLayout_file_select.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_target)
        self.formLayout_file_select.setWidget(1, QFormLayout.ItemRole.FieldRole, self.comboBox_target)

        # 数据列（多选）
        self.label_data_columns = QLabel(self.groupBox_file_select)
        self.label_data_columns.setObjectName(u"label_data_columns")
        self.label_data_columns.setText("数据列:")

        # 创建QListWidget作为多选下拉框
        self.listWidget_data_columns = QListWidget(self.groupBox_file_select)
        self.listWidget_data_columns.setObjectName(u"listWidget_data_columns")
        self.listWidget_data_columns.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.listWidget_data_columns.setEnabled(False)

        self.listWidget_data_columns.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.listWidget_data_columns.setMinimumHeight(150)

        self.listWidget_data_columns.setStyleSheet("""
            QListWidget {
                font-size: 13px;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e0f0ff;
                color: #333;
            }
        """)

        self.formLayout_file_select.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_data_columns)
        self.formLayout_file_select.setWidget(2, QFormLayout.ItemRole.FieldRole, self.listWidget_data_columns)

        # 识别CSV文件按钮
        self.horizontalLayout_identify_button = QHBoxLayout()
        self.horizontalLayout_identify_button.setObjectName(u"horizontalLayout_identify_button")
        self.horizontalLayout_identify_button.addStretch()

        self.btn_identify_csv = QPushButton(self.groupBox_file_select)
        self.btn_identify_csv.setObjectName(u"btn_identify_csv")
        self.btn_identify_csv.setText("识别CSV文件")
        self.btn_identify_csv.setFixedSize(120, 35)
        self.btn_identify_csv.setEnabled(False)  # 初始状态为禁用

        self.horizontalLayout_identify_button.addWidget(self.btn_identify_csv)
        self.formLayout_file_select.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_identify_button)

        self.verticalLayout_tab2.addWidget(self.groupBox_file_select, 1)  # 设置拉伸因子为1，占据剩余空间

        # ===== 第二个位置：预处理函数选择区域 =====
        self.groupBox_preprocess_func = QGroupBox(self.tab_process)
        self.groupBox_preprocess_func.setObjectName(u"groupBox_preprocess_func")
        self.groupBox_preprocess_func.setTitle("预处理函数选择")

        self.groupBox_preprocess_func.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        self.groupBox_preprocess_func.setFixedHeight(200)  # 固定高度

        # 预处理函数选择布局
        self.formLayout_preprocess_func = QFormLayout(self.groupBox_preprocess_func)
        self.formLayout_preprocess_func.setObjectName(u"formLayout_preprocess_func")
        self.formLayout_preprocess_func.setContentsMargins(15, 15, 15, 15)
        self.formLayout_preprocess_func.setSpacing(15)
        self.formLayout_preprocess_func.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.formLayout_preprocess_func.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # 滤波函数选择
        self.label_filter = QLabel(self.groupBox_preprocess_func)
        self.label_filter.setObjectName(u"label_filter")
        self.label_filter.setText("滤波函数:")

        self.comboBox_filter = QComboBox(self.groupBox_preprocess_func)
        self.comboBox_filter.setObjectName(u"comboBox_filter")
        self.comboBox_filter.addItems(
            ["--请选择--", "算术平均滤波法", "递推平均滤波法", "中位值平均滤波法", "一阶滞后滤波法",
             "加权递推平均滤波法", "消抖滤波法", "限幅消抖滤波法"])
        # 设置下拉框占满可用宽度
        self.comboBox_filter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.formLayout_preprocess_func.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_filter)
        self.formLayout_preprocess_func.setWidget(0, QFormLayout.ItemRole.FieldRole, self.comboBox_filter)

        # 值选择
        self.label_value = QLabel(self.groupBox_preprocess_func)
        self.label_value.setObjectName(u"label_value")
        self.label_value.setText("值选择:")

        self.comboBox_value = QComboBox(self.groupBox_preprocess_func)
        self.comboBox_value.setObjectName(u"comboBox_value")
        self.comboBox_value.addItems(["--请选择--", "平均值", "中位值", "众数", "极差", "最大值", "最大斜率"])
        # 设置下拉框占满可用宽度
        self.comboBox_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.formLayout_preprocess_func.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_value)
        self.formLayout_preprocess_func.setWidget(1, QFormLayout.ItemRole.FieldRole, self.comboBox_value)

        self.verticalLayout_tab2.addWidget(self.groupBox_preprocess_func)

        # ===== 底部：处理按钮区域 =====
        # 创建水平布局容器，用于放置按钮并居中
        self.horizontalLayout_process_buttons = QHBoxLayout()
        self.horizontalLayout_process_buttons.setObjectName(u"horizontalLayout_process_buttons")
        self.horizontalLayout_process_buttons.setContentsMargins(0, 10, 0, 10)

        # 数据预处理按钮
        self.btn_preprocess = QPushButton(self.tab_process)
        self.btn_preprocess.setObjectName(u"btn_preprocess")
        self.btn_preprocess.setText("数据预处理")
        self.btn_preprocess.setFixedSize(120, 35)

        # 数据合并按钮
        self.btn_merge = QPushButton(self.tab_process)
        self.btn_merge.setObjectName(u"btn_merge")
        self.btn_merge.setText("数据合并")
        self.btn_merge.setFixedSize(120, 35)

        # 添加按钮到布局中并居中
        self.horizontalLayout_process_buttons.addStretch()
        self.horizontalLayout_process_buttons.addWidget(self.btn_preprocess)
        self.horizontalLayout_process_buttons.addSpacing(20)
        self.horizontalLayout_process_buttons.addWidget(self.btn_merge)
        self.horizontalLayout_process_buttons.addStretch()

        # 将按钮布局添加到主垂直布局
        self.verticalLayout_tab2.addLayout(self.horizontalLayout_process_buttons)

        # ===== Tab: 算法模型 =====
        self.tab_results = QWidget()
        self.tab_results.setObjectName(u"tab_results")
        self.tabWidget.addTab(self.tab_results, "")

        self.verticalLayout_tab3 = QVBoxLayout(self.tab_results)
        self.verticalLayout_tab3.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_tab3.setSpacing(10)

        # ===== 第一部分：模型选择 =====
        self.groupBox_model = QGroupBox(self.tab_results)
        self.groupBox_model.setObjectName(u"groupBox_model")
        self.groupBox_model.setTitle("模型选择")

        self.groupBox_model.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        # 模型计算布局
        self.formLayout_model = QFormLayout(self.groupBox_model)
        self.formLayout_model.setObjectName(u"formLayout_model")
        self.formLayout_model.setContentsMargins(15, 15, 15, 15)
        self.formLayout_model.setSpacing(10)

        # 模型类型选择
        self.label_model_type = QLabel(self.groupBox_model)
        self.label_model_type.setObjectName(u"label_model_type")
        self.label_model_type.setText("模型类型:")

        self.comboBox_model_type = QComboBox(self.groupBox_model)
        self.comboBox_model_type.setObjectName(u"comboBox_model_type")
        self.comboBox_model_type.addItems(["--请选择--", "分类模型", "预测模型", "降维模型"])

        self.formLayout_model.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_model_type)
        self.formLayout_model.setWidget(0, QFormLayout.ItemRole.FieldRole, self.comboBox_model_type)

        # 具体模型选择
        self.label_model = QLabel(self.groupBox_model)
        self.label_model.setObjectName(u"label_model")
        self.label_model.setText("具体模型:")

        self.comboBox_model = QComboBox(self.groupBox_model)
        self.comboBox_model.setObjectName(u"comboBox_model")
        self.comboBox_model.addItems(["请先选择模型类型"])
        self.comboBox_model.setEnabled(False)

        self.formLayout_model.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_model)
        self.formLayout_model.setWidget(1, QFormLayout.ItemRole.FieldRole, self.comboBox_model)

        # 参数设置GroupBox
        self.groupBox_params = QGroupBox(self.groupBox_model)
        self.groupBox_params.setObjectName(u"groupBox_params")
        self.groupBox_params.setTitle("参数设置")

        self.verticalLayout_params = QVBoxLayout(self.groupBox_params)
        self.verticalLayout_params.setObjectName(u"verticalLayout_params")
        self.verticalLayout_params.setContentsMargins(10, 10, 10, 10)

        # 用于动态显示参数的网格布局
        self.gridLayout_params = QGridLayout()
        self.gridLayout_params.setObjectName(u"gridLayout_params")
        self.gridLayout_params.setHorizontalSpacing(15)
        self.gridLayout_params.setVerticalSpacing(10)

        # 初始占位标签
        self.label_params = QLabel(self.groupBox_params)
        self.label_params.setObjectName(u"label_params")
        self.label_params.setText("请先选择具体模型以显示参数设置")
        self.label_params.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_params.addWidget(self.label_params)
        self.verticalLayout_params.addLayout(self.gridLayout_params)

        self.formLayout_model.setWidget(2, QFormLayout.ItemRole.SpanningRole, self.groupBox_params)

        self.verticalLayout_tab3.addWidget(self.groupBox_model, 0)

        # ===== 第二部分：图片设置 =====
        self.groupBox_image_settings = QGroupBox(self.tab_results)
        self.groupBox_image_settings.setObjectName(u"groupBox_image_settings")
        self.groupBox_image_settings.setTitle("图片设置")

        self.groupBox_image_settings.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # 图片设置布局 - 使用垂直布局，这样更容易控制
        self.verticalLayout_image_settings = QVBoxLayout(self.groupBox_image_settings)
        self.verticalLayout_image_settings.setObjectName(u"verticalLayout_image_settings")
        self.verticalLayout_image_settings.setContentsMargins(15, 15, 15, 15)
        self.verticalLayout_image_settings.setSpacing(15)

        # 生成图片大小选项
        self.horizontalLayout_image_size = QHBoxLayout()
        self.horizontalLayout_image_size.setObjectName(u"horizontalLayout_image_size")
        self.horizontalLayout_image_size.setSpacing(10)

        self.label_image_size = QLabel(self.groupBox_image_settings)
        self.label_image_size.setObjectName(u"label_image_size")
        self.label_image_size.setText("生成图片大小:")
        self.label_image_size.setFixedWidth(130)  # 固定标签宽度

        self.comboBox_image_size = QComboBox(self.groupBox_image_settings)
        self.comboBox_image_size.setObjectName(u"comboBox_image_size")
        self.comboBox_image_size.addItems(["小", "较小", "适中", "较大", "大"])
        self.comboBox_image_size.setCurrentText("适中")

        self.horizontalLayout_image_size.addWidget(self.label_image_size)
        self.horizontalLayout_image_size.addWidget(self.comboBox_image_size)
        self.horizontalLayout_image_size.addStretch()

        self.verticalLayout_image_settings.addLayout(self.horizontalLayout_image_size)

        # 图片参数设置
        self.horizontalLayout_image_params = QHBoxLayout()
        self.horizontalLayout_image_params.setObjectName(u"horizontalLayout_image_params")
        self.horizontalLayout_image_params.setSpacing(10)

        self.label_image_params = QLabel(self.groupBox_image_settings)
        self.label_image_params.setObjectName(u"label_image_params")
        self.label_image_params.setText("图片参数:")
        self.label_image_params.setFixedWidth(120)

        self.textEdit_image_params = QTextEdit(self.groupBox_image_settings)
        self.textEdit_image_params.setObjectName(u"textEdit_image_params")
        self.textEdit_image_params.setPlaceholderText("请在此输入图片参数配置，如JSON格式的plot_config...")

        self.textEdit_image_params.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # 设置文本编辑框样式
        self.textEdit_image_params.setStyleSheet("""
            QTextEdit {
                font-size: 12px;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)

        self.horizontalLayout_image_params.addWidget(self.label_image_params)
        self.horizontalLayout_image_params.addWidget(self.textEdit_image_params)

        self.verticalLayout_image_settings.addLayout(self.horizontalLayout_image_params, 1)  # 拉伸因子为1，占据剩余空间

        self.verticalLayout_tab3.addWidget(self.groupBox_image_settings, 1)  # 拉伸因子为1，占据剩余高度

        # ===== 第三部分：计算按钮 =====
        self.horizontalLayout_calculate_button = QHBoxLayout()
        self.horizontalLayout_calculate_button.setObjectName(u"horizontalLayout_calculate_button")
        self.horizontalLayout_calculate_button.setContentsMargins(0, 10, 0, 0)

        self.btn_calculate = QPushButton(self.tab_results)
        self.btn_calculate.setObjectName(u"btn_calculate")
        self.btn_calculate.setText("计算")
        self.btn_calculate.setFixedSize(120, 35)

        self.horizontalLayout_calculate_button.addStretch()
        self.horizontalLayout_calculate_button.addWidget(self.btn_calculate)
        self.horizontalLayout_calculate_button.addStretch()

        self.verticalLayout_tab3.addLayout(self.horizontalLayout_calculate_button, 0)  # 拉伸因子为0，仅适合按钮高度

        # ===== Tab: 计算结果 =====
        self.tab_results_summary = QWidget()
        self.tab_results_summary.setObjectName(u"tab_results_summary")
        self.tabWidget.addTab(self.tab_results_summary, "")
        self.verticalLayout_tab4 = QVBoxLayout(self.tab_results_summary)
        self.verticalLayout_tab4.setObjectName(u"verticalLayout_tab4")
        self.verticalLayout_tab4.setContentsMargins(10, 10, 10, 10)

        self.groupBox_results = QGroupBox(self.tab_results_summary)
        self.groupBox_results.setObjectName(u"groupBox_results")
        self.groupBox_results.setTitle("计算结果概述")

        self.verticalLayout_results = QVBoxLayout(self.groupBox_results)
        self.verticalLayout_results.setObjectName(u"verticalLayout_results")
        self.verticalLayout_results.setContentsMargins(10, 10, 10, 10)

        # 创建文本编辑框显示结果
        self.textEdit_results = QTextEdit(self.groupBox_results)
        self.textEdit_results.setReadOnly(True)
        self.textEdit_results.setObjectName(u"textEdit_results")
        self.textEdit_results.setText("结果将显示在这里...")

        self.verticalLayout_results.addWidget(self.textEdit_results)

        # ===== 新的操作按钮区域 =====
        self.horizontalLayout_results_buttons = QHBoxLayout()
        self.horizontalLayout_results_buttons.setObjectName(u"horizontalLayout_results_buttons")
        self.horizontalLayout_results_buttons.setContentsMargins(0, 10, 0, 0)
        self.horizontalLayout_results_buttons.setSpacing(10)

        # 图片选择下拉框
        self.label_image_selector = QLabel(self.groupBox_results)
        self.label_image_selector.setObjectName(u"label_image_selector")
        self.label_image_selector.setText("图片列表:")
        self.label_image_selector.setFixedWidth(80)

        self.comboBox_image_selector = QComboBox(self.groupBox_results)
        self.comboBox_image_selector.setObjectName(u"comboBox_image_selector")
        self.comboBox_image_selector.addItems(["--请选择图片--"])
        self.comboBox_image_selector.setFixedWidth(200)

        # 显示图片按钮
        self.btn_show_image = QPushButton(self.groupBox_results)
        self.btn_show_image.setObjectName(u"btn_show_image")
        self.btn_show_image.setText("显示图片")
        self.btn_show_image.setFixedSize(100, 35)

        # 特征重要性另存为按钮
        self.btn_save_feature_importance = QPushButton(self.groupBox_results)
        self.btn_save_feature_importance.setObjectName(u"btn_save_feature_importance")
        self.btn_save_feature_importance.setText("特征重要性另存为")
        self.btn_save_feature_importance.setFixedSize(180, 35)

        # 模型信息另存为按钮
        self.btn_save_model_info = QPushButton(self.groupBox_results)
        self.btn_save_model_info.setObjectName(u"btn_save_model_info")
        self.btn_save_model_info.setText("模型信息另存为")
        self.btn_save_model_info.setFixedSize(150, 35)

        # 添加控件到水平布局
        self.horizontalLayout_results_buttons.addWidget(self.label_image_selector)
        self.horizontalLayout_results_buttons.addWidget(self.comboBox_image_selector)
        self.horizontalLayout_results_buttons.addWidget(self.btn_show_image)
        self.horizontalLayout_results_buttons.addWidget(self.btn_save_feature_importance)
        self.horizontalLayout_results_buttons.addWidget(self.btn_save_model_info)
        self.horizontalLayout_results_buttons.addStretch()

        self.verticalLayout_results.addLayout(self.horizontalLayout_results_buttons)

        self.verticalLayout_tab4.addWidget(self.groupBox_results)

        # 将TabWidget添加到右侧布局
        self.verticalLayout_right.addWidget(self.tabWidget)

        self.splitter.addWidget(self.right_widget)

        # 将分割器添加到主布局
        self.verticalLayout_9.addWidget(self.splitter)

        # 创建底部布局（包含状态栏和信息显示标签）
        self.horizontalLayout_bottom = QHBoxLayout()
        self.horizontalLayout_bottom.setObjectName(u"horizontalLayout_bottom")
        self.horizontalLayout_bottom.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout_bottom.setSpacing(10)

        # 创建状态栏
        self.statusBar = QLabel(Form)
        self.statusBar.setObjectName(u"statusBar")
        self.statusBar.setText("就绪")

        self.statusBar.setFrameShape(QFrame.Shape.NoFrame)
        self.statusBar.setFrameShadow(QFrame.Shadow.Plain)
        self.statusBar.setFixedHeight(25)

        font_status = QFont()
        font_status.setPointSize(10)
        self.statusBar.setFont(font_status)

        # 创建信息显示标签
        self.label_info = QLabel(Form)
        self.label_info.setObjectName(u"label_info")
        self.label_info.setFixedHeight(25)
        self.label_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        font_label = QFont()
        font_label.setPointSize(12)
        self.label_info.setFont(font_label)

        self.horizontalLayout_bottom.addWidget(self.statusBar, 1)
        self.horizontalLayout_bottom.addWidget(self.label_info, 0)

        self.verticalLayout_9.addLayout(self.horizontalLayout_bottom)

        total_width = 1165
        left_width = int(total_width * 0.3)
        right_width = total_width - left_width
        self.splitter.setSizes([left_width, right_width])

        self.retranslateUi(Form)

        self.tabWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"文件树与算法处理", None))
        self.btn_import.setText(QCoreApplication.translate("Form", u"导入文件夹", None))
        self.label_info.setText(QCoreApplication.translate("Form", u"已导入:0个文件夹|选中:0个文件", None))
        # 设置tab标签文本
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_process),
                                  QCoreApplication.translate("Form", u"数据预处理", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_results),
                                  QCoreApplication.translate("Form", u"算法模型", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_results_summary),
                                  QCoreApplication.translate("Form", u"结果", None))