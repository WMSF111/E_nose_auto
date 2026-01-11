# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'AlgModuleqayqJW.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSpacerItem, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1134, 703)
        self.verticalLayout_6 = QVBoxLayout(Form)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 9)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_19 = QLabel(self.layoutWidget)
        self.label_19.setObjectName(u"label_19")

        self.horizontalLayout_5.addWidget(self.label_19)

        self.lineEdit_Dir = QLineEdit(self.layoutWidget)
        self.lineEdit_Dir.setObjectName(u"lineEdit_Dir")
        self.lineEdit_Dir.setReadOnly(True)

        self.horizontalLayout_5.addWidget(self.lineEdit_Dir)

        self.btn_SelectDir = QPushButton(self.layoutWidget)
        self.btn_SelectDir.setObjectName(u"btn_SelectDir")

        self.horizontalLayout_5.addWidget(self.btn_SelectDir)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.table_FileList = QTableWidget(self.layoutWidget)
        if (self.table_FileList.columnCount() < 2):
            self.table_FileList.setColumnCount(2)
        self.table_FileList.setObjectName(u"table_FileList")
        self.table_FileList.setColumnCount(2)

        self.verticalLayout_2.addWidget(self.table_FileList)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_8)

        self.btn_Merge = QPushButton(self.layoutWidget)
        self.btn_Merge.setObjectName(u"btn_Merge")
        self.btn_Merge.setMinimumSize(QSize(0, 0))
        font = QFont()
        font.setPointSize(11)
        self.btn_Merge.setFont(font)

        self.horizontalLayout_9.addWidget(self.btn_Merge)


        self.verticalLayout_2.addLayout(self.horizontalLayout_9)

        self.groupBox_14 = QGroupBox(self.layoutWidget)
        self.groupBox_14.setObjectName(u"groupBox_14")
        self.verticalLayout = QVBoxLayout(self.groupBox_14)
        self.verticalLayout.setObjectName(u"verticalLayout")
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


        self.verticalLayout.addLayout(self.gridLayout_2)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_9)

        self.btn_Pre = QPushButton(self.groupBox_14)
        self.btn_Pre.setObjectName(u"btn_Pre")
        self.btn_Pre.setMinimumSize(QSize(0, 0))
        self.btn_Pre.setFont(font)

        self.horizontalLayout_8.addWidget(self.btn_Pre)


        self.verticalLayout.addLayout(self.horizontalLayout_8)


        self.verticalLayout_2.addWidget(self.groupBox_14)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_10)

        self.btn_AlgSelect = QPushButton(self.layoutWidget)
        self.btn_AlgSelect.setObjectName(u"btn_AlgSelect")
        self.btn_AlgSelect.setEnabled(True)
        self.btn_AlgSelect.setMinimumSize(QSize(0, 0))
        font1 = QFont()
        font1.setPointSize(11)
        font1.setBold(False)
        self.btn_AlgSelect.setFont(font1)
        self.btn_AlgSelect.setAutoFillBackground(False)
        self.btn_AlgSelect.setCheckable(False)
        self.btn_AlgSelect.setChecked(False)
        self.btn_AlgSelect.setAutoRepeat(False)
        self.btn_AlgSelect.setAutoDefault(False)
        self.btn_AlgSelect.setFlat(False)

        self.horizontalLayout_7.addWidget(self.btn_AlgSelect)


        self.verticalLayout_2.addLayout(self.horizontalLayout_7)

        self.splitter.addWidget(self.layoutWidget)
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
        self.splitter.addWidget(self.tabWidget)

        self.verticalLayout_6.addWidget(self.splitter)


        self.retranslateUi(Form)

        self.btn_AlgSelect.setDefault(False)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label_19.setText("")
        self.lineEdit_Dir.setPlaceholderText(QCoreApplication.translate("Form", u"\u539f\u6570\u636e\u6587\u4ef6\u76ee\u5f55", None))
        self.btn_SelectDir.setText(QCoreApplication.translate("Form", u"\u9009\u62e9", None))
#if QT_CONFIG(tooltip)
        self.btn_Merge.setToolTip(QCoreApplication.translate("Form", u"\u5728\u5217\u8868\u4e2d\u52fe\u9009\u8981\u5408\u5e76\u7684\u6587\u4ef6", None))
#endif // QT_CONFIG(tooltip)
        self.btn_Merge.setText(QCoreApplication.translate("Form", u"\u6570\u636e\u5408\u5e76", None))
        self.groupBox_14.setTitle(QCoreApplication.translate("Form", u"\u9884\u5904\u7406", None))
        self.label_20.setText(QCoreApplication.translate("Form", u"\u6ee4\u6ce2\u51fd\u6570", None))
        self.label_21.setText(QCoreApplication.translate("Form", u"\u503c\u9009\u62e9", None))
#if QT_CONFIG(tooltip)
        self.btn_Pre.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.btn_Pre.setText(QCoreApplication.translate("Form", u"\u6570\u636e\u9884\u5904\u7406", None))
        self.btn_AlgSelect.setText(QCoreApplication.translate("Form", u"\u6a21\u578b\u8ba1\u7b97", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"1. \u9009\u62e9\u539f\u6570\u636e\u6587\u4ef6\u76ee\u5f55", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"2. \u4ece\u6587\u4ef6\u5217\u8868\u4e2d\uff0c\u9009\u62e9\u539f\u6570\u636e\u6587\u4ef6\u8fdb\u884c\u5408\u5e76", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"3. \u8fdb\u884c\u6570\u636e\u9884\u5904\u7406", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"4. \u5728\u7b97\u6cd5\u6807\u7b7e\u9875\u4e2d\uff0c\u9009\u62e9\u7b97\u6cd5\uff0c\u8fdb\u884c\u8ba1\u7b97", None))
        self.label_8.setText(QCoreApplication.translate("Form", u"5. \u7b97\u6cd5\u7ed3\u679c\u5c55\u793a", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_welcome), QCoreApplication.translate("Form", u"\u4f7f\u7528\u8bf4\u660e", None))
    # retranslateUi

