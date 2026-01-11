# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'KNN.ui'
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
    QSizePolicy, QSpinBox, QToolButton, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(350, 300)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.verticalLayout.addWidget(self.pushButton_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.FilePath_lineEdit = QLineEdit(Form)
        self.FilePath_lineEdit.setObjectName(u"FilePath_lineEdit")

        self.horizontalLayout_2.addWidget(self.FilePath_lineEdit)

        self.toolButton = QToolButton(Form)
        self.toolButton.setObjectName(u"toolButton")

        self.horizontalLayout_2.addWidget(self.toolButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)

        self.label_5 = QLabel(Form)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)

        self.doubleSpinBox = QDoubleSpinBox(Form)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setMinimum(0.100000000000000)
        self.doubleSpinBox.setMaximum(0.500000000000000)
        self.doubleSpinBox.setSingleStep(0.100000000000000)
        self.doubleSpinBox.setValue(0.200000000000000)

        self.gridLayout.addWidget(self.doubleSpinBox, 0, 1, 1, 1)

        self.comboBox = QComboBox(Form)
        self.comboBox.setObjectName(u"comboBox")

        self.gridLayout.addWidget(self.comboBox, 3, 1, 1, 1)

        self.label = QLabel(Form)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.N_spinBox = QSpinBox(Form)
        self.N_spinBox.setObjectName(u"N_spinBox")
        self.N_spinBox.setMaximum(1000)
        self.N_spinBox.setValue(5)

        self.gridLayout.addWidget(self.N_spinBox, 2, 1, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 3)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.pushButton_2.setText(QCoreApplication.translate("Form", u"\u4f7f\u7528\u6a21\u578b", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u6587\u4ef6\u8def\u5f84", None))
        self.toolButton.setText(QCoreApplication.translate("Form", u"...", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"\u6700\u8fd1\u90bb\u7684\u7b97\u6cd5", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"\u90bb\u5c45\u6570", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u8bbe\u7f6e\u6d4b\u8bd5\u96c6\u5360\u6bd4(0.1-0.5)", None))
    # retranslateUi

