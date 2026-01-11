# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ChooseAndShow_2.ui'
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
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpinBox,
    QTabWidget, QTableView, QToolButton, QVBoxLayout,
    QWidget)


class Ui_Gragh_show(object):
    def setupUi(self, Gragh_show):
        if not Gragh_show.objectName():
            Gragh_show.setObjectName(u"Gragh_show")
        Gragh_show.resize(1246, 802)
        self.horizontalLayout_3 = QHBoxLayout(Gragh_show)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(Gragh_show)
        self.groupBox.setObjectName(u"groupBox")
        font = QFont()
        font.setBold(False)
        self.groupBox.setFont(font)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox_2 = QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.Folder_lineEdit = QLineEdit(self.groupBox_2)
        self.Folder_lineEdit.setObjectName(u"Folder_lineEdit")

        self.horizontalLayout_2.addWidget(self.Folder_lineEdit)

        self.Folder_Button = QToolButton(self.groupBox_2)
        self.Folder_Button.setObjectName(u"Folder_Button")

        self.horizontalLayout_2.addWidget(self.Folder_Button)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_2 = QLabel(self.groupBox_3)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_7.addWidget(self.label_2)

        self.Cleartime_spinBox = QSpinBox(self.groupBox_3)
        self.Cleartime_spinBox.setObjectName(u"Cleartime_spinBox")
        self.Cleartime_spinBox.setFocusPolicy(Qt.WheelFocus)
        self.Cleartime_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Cleartime_spinBox.setMaximum(256)
        self.Cleartime_spinBox.setValue(60)

        self.horizontalLayout_7.addWidget(self.Cleartime_spinBox)

        self.Dataclear_Button = QToolButton(self.groupBox_3)
        self.Dataclear_Button.setObjectName(u"Dataclear_Button")

        self.horizontalLayout_7.addWidget(self.Dataclear_Button)

        self.horizontalLayout_7.setStretch(0, 3)
        self.horizontalLayout_7.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_10 = QLabel(self.groupBox_3)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_6.addWidget(self.label_10)

        self.Volum_spinBox = QSpinBox(self.groupBox_3)
        self.Volum_spinBox.setObjectName(u"Volum_spinBox")
        self.Volum_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Volum_spinBox.setMinimum(0)
        self.Volum_spinBox.setMaximum(256)
        self.Volum_spinBox.setValue(1)
        self.Volum_spinBox.setDisplayIntegerBase(10)

        self.horizontalLayout_6.addWidget(self.Volum_spinBox)

        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_6.addWidget(self.label_9)

        self.Inputtime_spinBox = QSpinBox(self.groupBox_3)
        self.Inputtime_spinBox.setObjectName(u"Inputtime_spinBox")
        self.Inputtime_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Inputtime_spinBox.setMaximum(256)
        self.Inputtime_spinBox.setValue(60)

        self.horizontalLayout_6.addWidget(self.Inputtime_spinBox)

        self.Collectbegin_Button = QToolButton(self.groupBox_3)
        self.Collectbegin_Button.setObjectName(u"Collectbegin_Button")

        self.horizontalLayout_6.addWidget(self.Collectbegin_Button)

        self.horizontalLayout_6.setStretch(3, 2)

        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_5 = QLabel(self.groupBox_3)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_4.addWidget(self.label_5)

        self.Getchannel_spinBox = QSpinBox(self.groupBox_3)
        self.Getchannel_spinBox.setObjectName(u"Getchannel_spinBox")
        self.Getchannel_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Getchannel_spinBox.setMinimum(1)
        self.Getchannel_spinBox.setMaximum(8)
        self.Getchannel_spinBox.setValue(1)
        self.Getchannel_spinBox.setDisplayIntegerBase(10)

        self.horizontalLayout_4.addWidget(self.Getchannel_spinBox)

        self.label_7 = QLabel(self.groupBox_3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_4.addWidget(self.label_7)

        self.Gettep_spinBox = QDoubleSpinBox(self.groupBox_3)
        self.Gettep_spinBox.setObjectName(u"Gettep_spinBox")
        self.Gettep_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Gettep_spinBox.setDecimals(1)
        self.Gettep_spinBox.setMinimum(-1.000000000000000)
        self.Gettep_spinBox.setValue(-1.000000000000000)

        self.horizontalLayout_4.addWidget(self.Gettep_spinBox)

        self.Gettep_Button = QToolButton(self.groupBox_3)
        self.Gettep_Button.setObjectName(u"Gettep_Button")

        self.horizontalLayout_4.addWidget(self.Gettep_Button)

        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 1)
        self.horizontalLayout_4.setStretch(3, 2)
        self.horizontalLayout_4.setStretch(4, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_6 = QLabel(self.groupBox_3)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_8.addWidget(self.label_6)

        self.Heatchannel_spinBox = QSpinBox(self.groupBox_3)
        self.Heatchannel_spinBox.setObjectName(u"Heatchannel_spinBox")
        self.Heatchannel_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Heatchannel_spinBox.setMinimum(1)
        self.Heatchannel_spinBox.setMaximum(8)
        self.Heatchannel_spinBox.setValue(1)
        self.Heatchannel_spinBox.setDisplayIntegerBase(10)

        self.horizontalLayout_8.addWidget(self.Heatchannel_spinBox)

        self.label_8 = QLabel(self.groupBox_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_8.addWidget(self.label_8)

        self.Heattep_SpinBox_2 = QDoubleSpinBox(self.groupBox_3)
        self.Heattep_SpinBox_2.setObjectName(u"Heattep_SpinBox_2")
        self.Heattep_SpinBox_2.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Heattep_SpinBox_2.setDecimals(1)
        self.Heattep_SpinBox_2.setMinimum(-1.000000000000000)
        self.Heattep_SpinBox_2.setValue(50.000000000000000)

        self.horizontalLayout_8.addWidget(self.Heattep_SpinBox_2)

        self.Heat_Button = QToolButton(self.groupBox_3)
        self.Heat_Button.setObjectName(u"Heat_Button")

        self.horizontalLayout_8.addWidget(self.Heat_Button)

        self.horizontalLayout_8.setStretch(0, 2)
        self.horizontalLayout_8.setStretch(1, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_8)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.groupBox)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.gridLayout_2 = QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.Senser_stableView = QTableView(self.groupBox_4)
        self.Senser_stableView.setObjectName(u"Senser_stableView")
        self.Senser_stableView.setMinimumSize(QSize(226, 0))
        self.Senser_stableView.setMaximumSize(QSize(226, 16777215))

        self.gridLayout_2.addWidget(self.Senser_stableView, 0, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_4)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 3)
        self.verticalLayout.setStretch(2, 10)

        self.gridLayout.addWidget(self.groupBox, 0, 0, 2, 1)

        self.widget = QWidget(Gragh_show)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 0)
        self.Pause_Button = QPushButton(self.widget)
        self.Pause_Button.setObjectName(u"Pause_Button")
        self.Pause_Button.setMaximumSize(QSize(100, 32))

        self.horizontalLayout.addWidget(self.Pause_Button)

        self.Save_Button = QPushButton(self.widget)
        self.Save_Button.setObjectName(u"Save_Button")
        self.Save_Button.setMaximumSize(QSize(100, 32))

        self.horizontalLayout.addWidget(self.Save_Button)


        self.gridLayout.addWidget(self.widget, 0, 1, 1, 1)

        self.tabWidget = QTabWidget(Gragh_show)
        self.tabWidget.setObjectName(u"tabWidget")
        self.Line_Show = QWidget()
        self.Line_Show.setObjectName(u"Line_Show")
        self.gridLayout_5 = QGridLayout(self.Line_Show)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.Linegragh_Layout = QVBoxLayout()
        self.Linegragh_Layout.setObjectName(u"Linegragh_Layout")

        self.gridLayout_5.addLayout(self.Linegragh_Layout, 0, 0, 1, 1)

        self.tabWidget.addTab(self.Line_Show, "")

        self.gridLayout.addWidget(self.tabWidget, 1, 1, 1, 1)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 4)

        self.horizontalLayout_3.addLayout(self.gridLayout)


        self.retranslateUi(Gragh_show)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Gragh_show)
    # setupUi

    def retranslateUi(self, Gragh_show):
        Gragh_show.setWindowTitle(QCoreApplication.translate("Gragh_show", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("Gragh_show", u"\u5b9e\u9a8c\u4fe1\u606f", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Gragh_show", u"\u5b58\u50a8\u5730\u5740", None))
        self.Folder_Button.setText(QCoreApplication.translate("Gragh_show", u"...", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Gragh_show", u"\u91c7\u6837\u8bbe\u7f6e", None))
        self.label_2.setText(QCoreApplication.translate("Gragh_show", u"\u6e05\u6d17\u65f6\u957f(s)", None))
        self.Dataclear_Button.setText(QCoreApplication.translate("Gragh_show", u"\u6e05\u6d17", None))
        self.label_10.setText(QCoreApplication.translate("Gragh_show", u"\u91c7\u96c6\u4f53\u79ef(ml)", None))
        self.label_9.setText(QCoreApplication.translate("Gragh_show", u"\u65f6\u95f4(s)", None))
        self.Collectbegin_Button.setText(QCoreApplication.translate("Gragh_show", u"\u91c7\u6837", None))
        self.label_5.setText(QCoreApplication.translate("Gragh_show", u"\u83b7\u53d6\u901a\u9053(1-8)", None))
        self.label_7.setText(QCoreApplication.translate("Gragh_show", u"\u6e29\u5ea6", None))
        self.Gettep_Button.setText(QCoreApplication.translate("Gragh_show", u"\u83b7\u53d6", None))
        self.label_6.setText(QCoreApplication.translate("Gragh_show", u"\u52a0\u70ed\u901a\u9053(1-8)", None))
        self.label_8.setText(QCoreApplication.translate("Gragh_show", u"\u6e29\u5ea6", None))
        self.Heat_Button.setText(QCoreApplication.translate("Gragh_show", u"\u52a0\u70ed", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Gragh_show", u"\u4f20\u611f\u5668\u5217\u8868", None))
        self.Pause_Button.setText(QCoreApplication.translate("Gragh_show", u"\u6682\u505c\u91c7\u96c6", None))
        self.Save_Button.setText(QCoreApplication.translate("Gragh_show", u"\u4fdd\u5b58\u6570\u636e", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Line_Show), QCoreApplication.translate("Gragh_show", u"\u66f2\u7ebf\u56fe", None))
    # retranslateUi

