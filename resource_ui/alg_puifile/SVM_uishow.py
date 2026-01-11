# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SVM.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget)

class Ui_SVM_UI(object):
    def setupUi(self, SVM_UI):
        if not SVM_UI.objectName():
            SVM_UI.setObjectName(u"SVM_UI")
        SVM_UI.resize(350, 300)
        self.verticalLayout = QVBoxLayout(SVM_UI)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.pushButton_2 = QPushButton(SVM_UI)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.verticalLayout.addWidget(self.pushButton_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(SVM_UI)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.FilePath_lineEdit = QLineEdit(SVM_UI)
        self.FilePath_lineEdit.setObjectName(u"FilePath_lineEdit")

        self.horizontalLayout_2.addWidget(self.FilePath_lineEdit)

        self.toolButton = QToolButton(SVM_UI)
        self.toolButton.setObjectName(u"toolButton")

        self.horizontalLayout_2.addWidget(self.toolButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_5 = QLabel(SVM_UI)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)

        self.doubleSpinBox_C = QDoubleSpinBox(SVM_UI)
        self.doubleSpinBox_C.setObjectName(u"doubleSpinBox_C")
        self.doubleSpinBox_C.setDecimals(1)
        self.doubleSpinBox_C.setMinimum(0.000000000000000)
        self.doubleSpinBox_C.setMaximum(10000.000000000000000)
        self.doubleSpinBox_C.setSingleStep(1.000000000000000)
        self.doubleSpinBox_C.setValue(0.200000000000000)

        self.gridLayout.addWidget(self.doubleSpinBox_C, 2, 1, 1, 1)

        self.label = QLabel(SVM_UI)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.doubleSpinBox = QDoubleSpinBox(SVM_UI)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setMinimum(0.100000000000000)
        self.doubleSpinBox.setMaximum(0.500000000000000)
        self.doubleSpinBox.setSingleStep(0.100000000000000)
        self.doubleSpinBox.setValue(0.200000000000000)

        self.gridLayout.addWidget(self.doubleSpinBox, 0, 1, 1, 1)

        self.label_4 = QLabel(SVM_UI)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)

        self.comboBox = QComboBox(SVM_UI)
        self.comboBox.setObjectName(u"comboBox")

        self.gridLayout.addWidget(self.comboBox, 3, 1, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 3)

        self.retranslateUi(SVM_UI)

        QMetaObject.connectSlotsByName(SVM_UI)
    # setupUi

    def retranslateUi(self, SVM_UI):
        SVM_UI.setWindowTitle(QCoreApplication.translate("SVM_UI", u"Form", None))
        self.pushButton_2.setText(QCoreApplication.translate("SVM_UI", u"\u4f7f\u7528\u6a21\u578b", None))
        self.label_2.setText(QCoreApplication.translate("SVM_UI", u"\u6587\u4ef6\u8def\u5f84", None))
        self.toolButton.setText(QCoreApplication.translate("SVM_UI", u"...", None))
        self.label_5.setText(QCoreApplication.translate("SVM_UI", u"\u6b63\u5219\u5316\u53c2\u6570", None))
        self.doubleSpinBox_C.setSuffix("")
        self.label.setText(QCoreApplication.translate("SVM_UI", u"\u8bbe\u7f6e\u6d4b\u8bd5\u96c6\u5360\u6bd4(0.1-0.5)", None))
        self.label_4.setText(QCoreApplication.translate("SVM_UI", u"\u6838\u51fd\u6570\u7c7b\u578b", None))
    # retranslateUi

