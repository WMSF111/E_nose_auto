# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Alg_showriTkcR.ui'
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
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem, QTabWidget, QTextBrowser,
    QToolButton, QVBoxLayout, QWidget)

class Ui_Alg_show(object):
    def setupUi(self, Alg_show):
        if not Alg_show.objectName():
            Alg_show.setObjectName(u"Alg_show")
        Alg_show.resize(1140, 710)
        self.horizontalLayout_3 = QHBoxLayout(Alg_show)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.widget = QWidget(Alg_show)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 0)
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.Begin_Button = QPushButton(self.widget)
        self.Begin_Button.setObjectName(u"Begin_Button")
        self.Begin_Button.setMaximumSize(QSize(100, 32))

        self.horizontalLayout.addWidget(self.Begin_Button)


        self.gridLayout.addWidget(self.widget, 0, 1, 1, 1)

        self.tabWidget = QTabWidget(Alg_show)
        self.tabWidget.setObjectName(u"tabWidget")

        self.gridLayout.addWidget(self.tabWidget, 1, 1, 1, 1)

        self.groupBox = QGroupBox(Alg_show)
        self.groupBox.setObjectName(u"groupBox")
        font = QFont()
        font.setBold(False)
        self.groupBox.setFont(font)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.FilePath_lineEdit = QLineEdit(self.groupBox)
        self.FilePath_lineEdit.setObjectName(u"FilePath_lineEdit")

        self.horizontalLayout_2.addWidget(self.FilePath_lineEdit)

        self.toolButton = QToolButton(self.groupBox)
        self.toolButton.setObjectName(u"toolButton")

        self.horizontalLayout_2.addWidget(self.toolButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.groupBox_3 = QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_4 = QGridLayout(self.groupBox_3)
        self.gridLayout_4.setSpacing(3)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(-1, -1, -1, 3)
        self.Reg_ComboBox = QComboBox(self.groupBox_3)
        self.Reg_ComboBox.setObjectName(u"Reg_ComboBox")

        self.gridLayout_4.addWidget(self.Reg_ComboBox, 5, 1, 1, 1)

        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.gridLayout_4.addWidget(self.label_4, 1, 0, 1, 1)

        self.Classify_ComboBox = QComboBox(self.groupBox_3)
        self.Classify_ComboBox.setObjectName(u"Classify_ComboBox")

        self.gridLayout_4.addWidget(self.Classify_ComboBox, 7, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox_3)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.gridLayout_4.addWidget(self.label_3, 5, 0, 1, 1)

        self.label_2 = QLabel(self.groupBox_3)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.gridLayout_4.addWidget(self.label_2, 3, 0, 1, 1)

        self.label_5 = QLabel(self.groupBox_3)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_4.addWidget(self.label_5, 7, 0, 1, 1)

        self.Dimension_ComboBox = QComboBox(self.groupBox_3)
        self.Dimension_ComboBox.setObjectName(u"Dimension_ComboBox")

        self.gridLayout_4.addWidget(self.Dimension_ComboBox, 3, 1, 1, 1)

        self.Pre_Button = QPushButton(self.groupBox_3)
        self.Pre_Button.setObjectName(u"Pre_Button")

        self.gridLayout_4.addWidget(self.Pre_Button, 1, 1, 1, 1)

        self.gridLayout_4.setColumnStretch(0, 1)
        self.gridLayout_4.setColumnStretch(1, 2)

        self.verticalLayout.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.groupBox)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.gridLayout_2 = QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.textBrowser = QTextBrowser(self.groupBox_4)
        self.textBrowser.setObjectName(u"textBrowser")

        self.gridLayout_2.addWidget(self.textBrowser, 0, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_4)

        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 4)

        self.gridLayout.addWidget(self.groupBox, 0, 0, 2, 1)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 4)

        self.horizontalLayout_3.addLayout(self.gridLayout)


        self.retranslateUi(Alg_show)

        self.tabWidget.setCurrentIndex(-1)


        QMetaObject.connectSlotsByName(Alg_show)
    # setupUi

    def retranslateUi(self, Alg_show):
        Alg_show.setWindowTitle(QCoreApplication.translate("Alg_show", u"Form", None))
        self.Begin_Button.setText(QCoreApplication.translate("Alg_show", u"\u5f00\u59cb\u5904\u7406", None))
        self.groupBox.setTitle(QCoreApplication.translate("Alg_show", u"\u5b9e\u9a8c\u4fe1\u606f", None))
        self.label.setText(QCoreApplication.translate("Alg_show", u"\u6587\u4ef6\u8def\u5f84", None))
        self.toolButton.setText(QCoreApplication.translate("Alg_show", u"...", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Alg_show", u"\u7b97\u6cd5\u9009\u62e9", None))
        self.label_4.setText(QCoreApplication.translate("Alg_show", u"\u9884\u5904\u7406", None))
        self.label_3.setText(QCoreApplication.translate("Alg_show", u"\u56de\u5f52\u9884\u6d4b", None))
        self.label_2.setText(QCoreApplication.translate("Alg_show", u"\u964d\u7ef4\u7b97\u6cd5", None))
        self.label_5.setText(QCoreApplication.translate("Alg_show", u"\u5206\u7c7b\u7b97\u6cd5", None))
        self.Pre_Button.setText(QCoreApplication.translate("Alg_show", u"\u7b97\u6cd5\u9009\u62e9", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Alg_show", u"\u6570\u636e\u9009\u62e9", None))
    # retranslateUi

