# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Regression.ui'
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
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget)

class Ui_Regression(object):
    def setupUi(self, Regression_uishow):
        if not Regression_uishow.objectName():
            Regression_uishow.setObjectName(u"Regression_uishow")
        Regression_uishow.resize(306, 238)
        self.verticalLayout = QVBoxLayout(Regression_uishow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.pushButton_2 = QPushButton(Regression_uishow)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.verticalLayout.addWidget(self.pushButton_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(Regression_uishow)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.FilePath_lineEdit = QLineEdit(Regression_uishow)
        self.FilePath_lineEdit.setObjectName(u"FilePath_lineEdit")

        self.horizontalLayout_2.addWidget(self.FilePath_lineEdit)

        self.toolButton = QToolButton(Regression_uishow)
        self.toolButton.setObjectName(u"toolButton")

        self.horizontalLayout_2.addWidget(self.toolButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(Regression_uishow)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.doubleSpinBox = QDoubleSpinBox(Regression_uishow)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setMinimum(0.100000000000000)
        self.doubleSpinBox.setMaximum(0.500000000000000)
        self.doubleSpinBox.setSingleStep(0.100000000000000)
        self.doubleSpinBox.setValue(0.200000000000000)

        self.horizontalLayout.addWidget(self.doubleSpinBox)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 4)

        self.retranslateUi(Regression_uishow)

        QMetaObject.connectSlotsByName(Regression_uishow)
    # setupUi

    def retranslateUi(self, Regression_uishow):
        Regression_uishow.setWindowTitle(QCoreApplication.translate("Regression_uishow", u"Form", None))
        self.pushButton_2.setText(QCoreApplication.translate("Regression_uishow", u"\u4f7f\u7528\u6a21\u578b", None))
        self.label_2.setText(QCoreApplication.translate("Regression_uishow", u"\u6587\u4ef6\u8def\u5f84", None))
        self.toolButton.setText(QCoreApplication.translate("Regression_uishow", u"...", None))
        self.label.setText(QCoreApplication.translate("Regression_uishow", u"\u8bbe\u7f6e\u6d4b\u8bd5\u96c6\u5360\u6bd4(0.1-0.5)", None))
    # retranslateUi

