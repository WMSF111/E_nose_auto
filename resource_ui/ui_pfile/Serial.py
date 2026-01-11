# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Serial.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
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
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QTextBrowser, QVBoxLayout,
    QWidget)

class Ui_Serial(object):
    def setupUi(self, Serial):
        if not Serial.objectName():
            Serial.setObjectName(u"Serial")
        Serial.resize(641, 490)
        self.gridLayout = QGridLayout(Serial)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox = QGroupBox(Serial)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.formLayout = QFormLayout(self.groupBox)
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.CheckButton = QPushButton(self.groupBox)
        self.CheckButton.setObjectName(u"CheckButton")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.CheckButton)

        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_5)

        self.serialComboBox = QComboBox(self.groupBox)
        self.serialComboBox.setObjectName(u"serialComboBox")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.serialComboBox)

        self.statues = QLabel(self.groupBox)
        self.statues.setObjectName(u"statues")
        self.statues.setTextFormat(Qt.AutoText)

        self.formLayout.setWidget(2, QFormLayout.SpanningRole, self.statues)

        self.label_15 = QLabel(self.groupBox)
        self.label_15.setObjectName(u"label_15")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_15)

        self.serialComboBox2 = QComboBox(self.groupBox)
        self.serialComboBox2.setObjectName(u"serialComboBox2")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.serialComboBox2)

        self.statues_3 = QLabel(self.groupBox)
        self.statues_3.setObjectName(u"statues_3")
        self.statues_3.setTextFormat(Qt.AutoText)

        self.formLayout.setWidget(4, QFormLayout.SpanningRole, self.statues_3)

        self.connectButton = QPushButton(self.groupBox)
        self.connectButton.setObjectName(u"connectButton")

        self.formLayout.setWidget(5, QFormLayout.SpanningRole, self.connectButton)


        self.verticalLayout.addWidget(self.groupBox)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.sendEdit = QLineEdit(Serial)
        self.sendEdit.setObjectName(u"sendEdit")

        self.horizontalLayout_2.addWidget(self.sendEdit)

        self.sendButton = QPushButton(Serial)
        self.sendButton.setObjectName(u"sendButton")

        self.horizontalLayout_2.addWidget(self.sendButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.saveButton = QPushButton(Serial)
        self.saveButton.setObjectName(u"saveButton")

        self.horizontalLayout_3.addWidget(self.saveButton)

        self.clearButton = QPushButton(Serial)
        self.clearButton.setObjectName(u"clearButton")

        self.horizontalLayout_3.addWidget(self.clearButton)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.verticalLayout.setStretch(0, 7)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.tb = QTextBrowser(Serial)
        self.tb.setObjectName(u"tb")

        self.horizontalLayout.addWidget(self.tb)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)

        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)


        self.retranslateUi(Serial)

        QMetaObject.connectSlotsByName(Serial)
    # setupUi

    def retranslateUi(self, Serial):
        Serial.setWindowTitle(QCoreApplication.translate("Serial", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("Serial", u"\u4e32\u53e3\u8bbe\u7f6e", None))
        self.label.setText(QCoreApplication.translate("Serial", u"\u4e32\u53e3\u68c0\u6d4b", None))
        self.CheckButton.setText(QCoreApplication.translate("Serial", u"\u5f00\u59cb\u68c0\u6d4b", None))
        self.label_5.setText(QCoreApplication.translate("Serial", u"<html><head/><body><p align=\"center\">\u4fe1\u53f7\u4e32\u53e3</p></body></html>", None))
        self.statues.setText(QCoreApplication.translate("Serial", u"<html><head/><body><p align=\"center\"><span style=\" color:#ff0000;\"><br/></span></p></body></html>", None))
        self.label_15.setText(QCoreApplication.translate("Serial", u"<html><head/><body><p align=\"center\">\u63a7\u5236\u4e32\u53e3</p></body></html>", None))
        self.statues_3.setText(QCoreApplication.translate("Serial", u"<html><head/><body><p align=\"center\"><span style=\" color:#ff0000;\"><br/></span></p></body></html>", None))
        self.connectButton.setText(QCoreApplication.translate("Serial", u"\u8fde\u63a5", None))
        self.sendButton.setText(QCoreApplication.translate("Serial", u"\u53d1\u9001", None))
        self.saveButton.setText(QCoreApplication.translate("Serial", u"\u4fdd\u5b58\u4e3a\u6587\u4ef6", None))
        self.clearButton.setText(QCoreApplication.translate("Serial", u"\u6e05\u7a7a\u8f93\u51fa", None))
    # retranslateUi

