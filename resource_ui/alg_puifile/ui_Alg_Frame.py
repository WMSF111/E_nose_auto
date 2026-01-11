# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Alg_FrameNoKRRP.ui'
##
## Created by: Qt User Interface Compiler version 6.7.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
    QSizePolicy, QSpacerItem, QSplitter, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1165, 866)
        self.verticalLayout_9 = QVBoxLayout(Form)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.widget = QWidget(self.splitter)
        self.widget.setObjectName(u"widget")
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_19 = QLabel(self.widget)
        self.label_19.setObjectName(u"label_19")

        self.horizontalLayout_5.addWidget(self.label_19)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer)

        self.btn_import = QPushButton(self.widget)
        self.btn_import.setObjectName(u"btn_import")

        self.horizontalLayout_5.addWidget(self.btn_import)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.treeWidget = QTreeWidget(self.widget)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.treeWidget.setHeaderItem(__qtreewidgetitem)
        self.treeWidget.setObjectName(u"treeWidget")
        self.treeWidget.setColumnCount(1)

        self.verticalLayout.addWidget(self.treeWidget)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_12)

        self.btn_unchecked_all = QPushButton(self.widget)
        self.btn_unchecked_all.setObjectName(u"btn_unchecked_all")

        self.horizontalLayout_10.addWidget(self.btn_unchecked_all)

        self.btn_clear_work_path = QPushButton(self.widget)
        self.btn_clear_work_path.setObjectName(u"btn_clear_work_path")

        self.horizontalLayout_10.addWidget(self.btn_clear_work_path)


        self.verticalLayout.addLayout(self.horizontalLayout_10)

        self.splitter.addWidget(self.widget)
        self.tabWidget = QTabWidget(self.splitter)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setTabsClosable(True)
        self.tab_welcome = QWidget()
        self.tab_welcome.setObjectName(u"tab_welcome")
        self.verticalLayout_11 = QVBoxLayout(self.tab_welcome)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalSpacer_2 = QSpacerItem(20, 250, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_2)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_6)

        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.label_4 = QLabel(self.tab_welcome)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setStyleSheet(u"color: rgb(100, 100, 100);")

        self.verticalLayout_10.addWidget(self.label_4)

        self.label_5 = QLabel(self.tab_welcome)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setStyleSheet(u"color: rgb(100, 100, 100);")

        self.verticalLayout_10.addWidget(self.label_5)

        self.label_6 = QLabel(self.tab_welcome)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setStyleSheet(u"color: rgb(100, 100, 100);")

        self.verticalLayout_10.addWidget(self.label_6)

        self.label_7 = QLabel(self.tab_welcome)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setStyleSheet(u"color: rgb(100, 100, 100);")

        self.verticalLayout_10.addWidget(self.label_7)

        self.label_8 = QLabel(self.tab_welcome)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setStyleSheet(u"color: rgb(100, 100, 100);")

        self.verticalLayout_10.addWidget(self.label_8)


        self.horizontalLayout_6.addLayout(self.verticalLayout_10)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_7)


        self.verticalLayout_11.addLayout(self.horizontalLayout_6)

        self.verticalSpacer_3 = QSpacerItem(20, 250, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_3)

        self.tabWidget.addTab(self.tab_welcome, "")
        self.tab_pre = QWidget()
        self.tab_pre.setObjectName(u"tab_pre")
        self.verticalLayout_5 = QVBoxLayout(self.tab_pre)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.groupBox_14 = QGroupBox(self.tab_pre)
        self.groupBox_14.setObjectName(u"groupBox_14")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_14)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_20 = QLabel(self.groupBox_14)
        self.label_20.setObjectName(u"label_20")

        self.gridLayout_2.addWidget(self.label_20, 0, 0, 1, 1)

        self.cBox_FilterFun = QComboBox(self.groupBox_14)
        self.cBox_FilterFun.setObjectName(u"cBox_FilterFun")

        self.gridLayout_2.addWidget(self.cBox_FilterFun, 0, 1, 1, 1)

        self.label_21 = QLabel(self.groupBox_14)
        self.label_21.setObjectName(u"label_21")

        self.gridLayout_2.addWidget(self.label_21, 1, 0, 1, 1)

        self.cBox_ValSelect = QComboBox(self.groupBox_14)
        self.cBox_ValSelect.setObjectName(u"cBox_ValSelect")

        self.gridLayout_2.addWidget(self.cBox_ValSelect, 1, 1, 1, 1)


        self.verticalLayout_2.addLayout(self.gridLayout_2)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.btn_Merge = QPushButton(self.groupBox_14)
        self.btn_Merge.setObjectName(u"btn_Merge")
        self.btn_Merge.setMinimumSize(QSize(0, 0))
        font = QFont()
        font.setPointSize(11)
        self.btn_Merge.setFont(font)

        self.horizontalLayout_8.addWidget(self.btn_Merge)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_9)

        self.btn_Pre = QPushButton(self.groupBox_14)
        self.btn_Pre.setObjectName(u"btn_Pre")
        self.btn_Pre.setMinimumSize(QSize(0, 0))
        self.btn_Pre.setFont(font)

        self.horizontalLayout_8.addWidget(self.btn_Pre)


        self.verticalLayout_2.addLayout(self.horizontalLayout_8)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)


        self.verticalLayout_4.addLayout(self.horizontalLayout)


        self.verticalLayout_5.addWidget(self.groupBox_14)

        self.groupBox_15 = QGroupBox(self.tab_pre)
        self.groupBox_15.setObjectName(u"groupBox_15")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_15)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.pTextEdit_pre = QPlainTextEdit(self.groupBox_15)
        self.pTextEdit_pre.setObjectName(u"pTextEdit_pre")

        self.verticalLayout_3.addWidget(self.pTextEdit_pre)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_11)

        self.btn_clear_pre = QPushButton(self.groupBox_15)
        self.btn_clear_pre.setObjectName(u"btn_clear_pre")

        self.horizontalLayout_9.addWidget(self.btn_clear_pre)


        self.verticalLayout_3.addLayout(self.horizontalLayout_9)


        self.verticalLayout_5.addWidget(self.groupBox_15)

        self.verticalSpacer = QSpacerItem(20, 324, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer)

        self.tabWidget.addTab(self.tab_pre, "")
        self.tab_process = QWidget()
        self.tab_process.setObjectName(u"tab_process")
        self.verticalLayout_8 = QVBoxLayout(self.tab_process)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.groupBox_2 = QGroupBox(self.tab_process)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_6 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(9)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.cBox_AlgGroup = QComboBox(self.groupBox_2)
        self.cBox_AlgGroup.setObjectName(u"cBox_AlgGroup")
        self.cBox_AlgGroup.setMinimumSize(QSize(120, 0))

        self.horizontalLayout_2.addWidget(self.cBox_AlgGroup)

        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")
        self.label.setStyleSheet(u"color: rgb(162, 162, 162);")

        self.horizontalLayout_2.addWidget(self.label)

        self.cBox_AlgName = QComboBox(self.groupBox_2)
        self.cBox_AlgName.setObjectName(u"cBox_AlgName")
        self.cBox_AlgName.setMinimumSize(QSize(120, 0))

        self.horizontalLayout_2.addWidget(self.cBox_AlgName)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)


        self.verticalLayout_6.addLayout(self.horizontalLayout_2)

        self.groupBox_5 = QGroupBox(self.groupBox_2)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.horizontalLayout_4 = QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")

        self.horizontalLayout_4.addLayout(self.formLayout)

        self.horizontalSpacer_4 = QSpacerItem(492, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_4)


        self.verticalLayout_6.addWidget(self.groupBox_5)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_3 = QSpacerItem(588, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)

        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_3.addWidget(self.label_2)

        self.rbtn_smallMin = QRadioButton(self.groupBox_2)
        self.rbtn_smallMin.setObjectName(u"rbtn_smallMin")

        self.horizontalLayout_3.addWidget(self.rbtn_smallMin)

        self.rbtn_small = QRadioButton(self.groupBox_2)
        self.rbtn_small.setObjectName(u"rbtn_small")

        self.horizontalLayout_3.addWidget(self.rbtn_small)

        self.rbtn_middle = QRadioButton(self.groupBox_2)
        self.rbtn_middle.setObjectName(u"rbtn_middle")
        self.rbtn_middle.setChecked(True)

        self.horizontalLayout_3.addWidget(self.rbtn_middle)

        self.rbtn_larger = QRadioButton(self.groupBox_2)
        self.rbtn_larger.setObjectName(u"rbtn_larger")

        self.horizontalLayout_3.addWidget(self.rbtn_larger)

        self.rbtn_largerMax = QRadioButton(self.groupBox_2)
        self.rbtn_largerMax.setObjectName(u"rbtn_largerMax")

        self.horizontalLayout_3.addWidget(self.rbtn_largerMax)

        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(20, 0))

        self.horizontalLayout_3.addWidget(self.label_3)

        self.btn_run = QPushButton(self.groupBox_2)
        self.btn_run.setObjectName(u"btn_run")
        font1 = QFont()
        font1.setPointSize(11)
        font1.setBold(False)
        self.btn_run.setFont(font1)

        self.horizontalLayout_3.addWidget(self.btn_run)


        self.verticalLayout_6.addLayout(self.horizontalLayout_3)


        self.verticalLayout_8.addWidget(self.groupBox_2)

        self.progressBar = QProgressBar(self.tab_process)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setMinimumSize(QSize(0, 20))
        self.progressBar.setMaximumSize(QSize(16777215, 20))
        self.progressBar.setMaximum(0)
        self.progressBar.setValue(-1)
        self.progressBar.setTextVisible(False)

        self.verticalLayout_8.addWidget(self.progressBar)

        self.groupBox_3 = QGroupBox(self.tab_process)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_7 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.pTextEdit_model = QPlainTextEdit(self.groupBox_3)
        self.pTextEdit_model.setObjectName(u"pTextEdit_model")

        self.verticalLayout_7.addWidget(self.pTextEdit_model)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_8)

        self.btn_scree_plot = QPushButton(self.groupBox_3)
        self.btn_scree_plot.setObjectName(u"btn_scree_plot")

        self.horizontalLayout_7.addWidget(self.btn_scree_plot)

        self.btn_scatter_plot = QPushButton(self.groupBox_3)
        self.btn_scatter_plot.setObjectName(u"btn_scatter_plot")

        self.horizontalLayout_7.addWidget(self.btn_scatter_plot)

        self.btn_source_data = QPushButton(self.groupBox_3)
        self.btn_source_data.setObjectName(u"btn_source_data")

        self.horizontalLayout_7.addWidget(self.btn_source_data)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_10)


        self.verticalLayout_7.addLayout(self.horizontalLayout_7)


        self.verticalLayout_8.addWidget(self.groupBox_3)

        self.verticalSpacer_4 = QSpacerItem(20, 264, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_8.addItem(self.verticalSpacer_4)

        self.tabWidget.addTab(self.tab_process, "")
        self.splitter.addWidget(self.tabWidget)

        self.verticalLayout_9.addWidget(self.splitter)


        self.retranslateUi(Form)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label_19.setText("")
        self.btn_import.setText(QCoreApplication.translate("Form", u"\u5bfc\u5165\u6570\u636e\u76ee\u5f55", None))
        self.btn_unchecked_all.setText(QCoreApplication.translate("Form", u"\u53d6\u6d88\u52fe\u9009", None))
        self.btn_clear_work_path.setText(QCoreApplication.translate("Form", u" \u6e05\u7a7a\u5de5\u4f5c\u76ee\u5f55", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"1. \u9009\u62e9\u539f\u6570\u636e\u6587\u4ef6\u76ee\u5f55", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"2. \u4ece\u6587\u4ef6\u5217\u8868\u4e2d\uff0c\u9009\u62e9\u539f\u6570\u636e\u6587\u4ef6\u8fdb\u884c\u5408\u5e76", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"3. \u8fdb\u884c\u6570\u636e\u9884\u5904\u7406", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"4. \u5728\u7b97\u6cd5\u6807\u7b7e\u9875\u4e2d\uff0c\u9009\u62e9\u7b97\u6cd5\uff0c\u8fdb\u884c\u8ba1\u7b97", None))
        self.label_8.setText(QCoreApplication.translate("Form", u"5. \u7b97\u6cd5\u7ed3\u679c\u5c55\u793a", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_welcome), QCoreApplication.translate("Form", u"\u4f7f\u7528\u8bf4\u660e", None))
        self.groupBox_14.setTitle(QCoreApplication.translate("Form", u"\u9884\u5904\u7406", None))
        self.label_20.setText(QCoreApplication.translate("Form", u"\u6ee4\u6ce2\u51fd\u6570", None))
        self.label_21.setText(QCoreApplication.translate("Form", u"\u503c\u9009\u62e9", None))
#if QT_CONFIG(tooltip)
        self.btn_Merge.setToolTip(QCoreApplication.translate("Form", u"\u5728\u5217\u8868\u4e2d\u52fe\u9009\u8981\u5408\u5e76\u7684\u6587\u4ef6", None))
#endif // QT_CONFIG(tooltip)
        self.btn_Merge.setText(QCoreApplication.translate("Form", u"\u6570\u636e\u5408\u5e76", None))
#if QT_CONFIG(tooltip)
        self.btn_Pre.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.btn_Pre.setText(QCoreApplication.translate("Form", u"\u6570\u636e\u9884\u5904\u7406", None))
        self.groupBox_15.setTitle(QCoreApplication.translate("Form", u"\u9884\u5904\u7406\u7ed3\u679c", None))
        self.btn_clear_pre.setText(QCoreApplication.translate("Form", u"\u6e05\u7a7a", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_pre), QCoreApplication.translate("Form", u"\u6570\u636e\u9884\u5904\u7406", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"\u7b97\u6cd5\u9009\u62e9", None))
        self.label.setText(QCoreApplication.translate("Form", u">>", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Form", u"\u7b97\u6cd5\u53c2\u6570\u8bbe\u7f6e", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u51fa\u56fe\u5c3a\u5bf8:", None))
        self.rbtn_smallMin.setText(QCoreApplication.translate("Form", u"\u5c0f", None))
        self.rbtn_small.setText(QCoreApplication.translate("Form", u"\u8f83\u5c0f", None))
        self.rbtn_middle.setText(QCoreApplication.translate("Form", u"\u9002\u4e2d", None))
        self.rbtn_larger.setText(QCoreApplication.translate("Form", u"\u8f83\u5927", None))
        self.rbtn_largerMax.setText(QCoreApplication.translate("Form", u"\u5927", None))
        self.label_3.setText("")
        self.btn_run.setText(QCoreApplication.translate("Form", u"\u8ba1\u7b97", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Form", u"\u8ba1\u7b97\u7ed3\u679c\u6982\u8ff0", None))
        self.pTextEdit_model.setPlainText("")
        self.btn_scree_plot.setText(QCoreApplication.translate("Form", u"\u663e\u793a\u788e\u77f3\u56fe", None))
        self.btn_scatter_plot.setText(QCoreApplication.translate("Form", u"\u663e\u793a\u6563\u70b9\u56fe", None))
        self.btn_source_data.setText(QCoreApplication.translate("Form", u"\u663e\u793a\u539f\u6570\u636e", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_process), QCoreApplication.translate("Form", u"\u6a21\u578b\u8ba1\u7b97", None))
    # retranslateUi

