# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ChooseAndShow_1.ui'
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
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSpinBox, QTabWidget,
    QTableView, QToolButton, QVBoxLayout, QWidget)

class Ui_Gragh_show(object):
    def setupUi(self, Gragh_show):
        if not Gragh_show.objectName():
            Gragh_show.setObjectName(u"Gragh_show")
        Gragh_show.resize(997, 662)
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
        self.verticalLayout.setContentsMargins(3, -1, 3, -1)
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
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_2 = QLabel(self.groupBox_3)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_7.addWidget(self.label_2)

        self.Cleartime_spinBox = QSpinBox(self.groupBox_3)
        self.Cleartime_spinBox.setObjectName(u"Cleartime_spinBox")
        font1 = QFont()
        font1.setFamilies([u"AcadEref"])
        font1.setPointSize(8)
        font1.setBold(False)
        self.Cleartime_spinBox.setFont(font1)
        self.Cleartime_spinBox.setFocusPolicy(Qt.WheelFocus)
        self.Cleartime_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Cleartime_spinBox.setMinimum(10)
        self.Cleartime_spinBox.setMaximum(256)
        self.Cleartime_spinBox.setValue(10)

        self.horizontalLayout_7.addWidget(self.Cleartime_spinBox)

        self.label_34 = QLabel(self.groupBox_3)
        self.label_34.setObjectName(u"label_34")
        self.label_34.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_7.addWidget(self.label_34)

        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_7.addWidget(self.label_4)

        self.Sample_spinBox = QSpinBox(self.groupBox_3)
        self.Sample_spinBox.setObjectName(u"Sample_spinBox")
        self.Sample_spinBox.setFont(font1)
        self.Sample_spinBox.setFocusPolicy(Qt.WheelFocus)
        self.Sample_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Sample_spinBox.setMinimum(0)
        self.Sample_spinBox.setMaximum(256)
        self.Sample_spinBox.setValue(10)

        self.horizontalLayout_7.addWidget(self.Sample_spinBox)

        self.label_35 = QLabel(self.groupBox_3)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_7.addWidget(self.label_35)

        self.horizontalLayout_7.setStretch(0, 2)
        self.horizontalLayout_7.setStretch(1, 3)
        self.horizontalLayout_7.setStretch(2, 1)
        self.horizontalLayout_7.setStretch(3, 2)
        self.horizontalLayout_7.setStretch(4, 3)
        self.horizontalLayout_7.setStretch(5, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_8.addWidget(self.label_9)

        self.Basetime_spinBox = QSpinBox(self.groupBox_3)
        self.Basetime_spinBox.setObjectName(u"Basetime_spinBox")
        self.Basetime_spinBox.setFont(font1)
        self.Basetime_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Basetime_spinBox.setMinimum(0)
        self.Basetime_spinBox.setMaximum(256)
        self.Basetime_spinBox.setValue(10)
        self.Basetime_spinBox.setDisplayIntegerBase(10)

        self.horizontalLayout_8.addWidget(self.Basetime_spinBox)

        self.label_36 = QLabel(self.groupBox_3)
        self.label_36.setObjectName(u"label_36")
        self.label_36.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_8.addWidget(self.label_36)

        self.label_14 = QLabel(self.groupBox_3)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_8.addWidget(self.label_14)

        self.Worktime_spinBox = QSpinBox(self.groupBox_3)
        self.Worktime_spinBox.setObjectName(u"Worktime_spinBox")
        self.Worktime_spinBox.setFont(font1)
        self.Worktime_spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Worktime_spinBox.setMinimum(1)
        self.Worktime_spinBox.setMaximum(256)
        self.Worktime_spinBox.setValue(1)
        self.Worktime_spinBox.setDisplayIntegerBase(10)

        self.horizontalLayout_8.addWidget(self.Worktime_spinBox)

        self.label_38 = QLabel(self.groupBox_3)
        self.label_38.setObjectName(u"label_38")
        self.label_38.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_8.addWidget(self.label_38)

        self.horizontalLayout_8.setStretch(0, 2)
        self.horizontalLayout_8.setStretch(1, 3)
        self.horizontalLayout_8.setStretch(2, 1)
        self.horizontalLayout_8.setStretch(3, 2)
        self.horizontalLayout_8.setStretch(4, 3)
        self.horizontalLayout_8.setStretch(5, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.Auto_Button = QPushButton(self.groupBox_3)
        self.Auto_Button.setObjectName(u"Auto_Button")
        self.Auto_Button.setMaximumSize(QSize(200, 32))

        self.horizontalLayout_4.addWidget(self.Auto_Button)

        self.InitPos_Button = QPushButton(self.groupBox_3)
        self.InitPos_Button.setObjectName(u"InitPos_Button")
        self.InitPos_Button.setMaximumSize(QSize(200, 32))

        self.horizontalLayout_4.addWidget(self.InitPos_Button)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.groupBox)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.gridLayout_2 = QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.Senser_stableView = QTableView(self.groupBox_4)
        self.Senser_stableView.setObjectName(u"Senser_stableView")
        self.Senser_stableView.setMinimumSize(QSize(226, 0))
        self.Senser_stableView.setMaximumSize(QSize(500, 16777215))
        font2 = QFont()
        font2.setPointSize(8)
        font2.setBold(False)
        self.Senser_stableView.setFont(font2)

        self.gridLayout_2.addWidget(self.Senser_stableView, 0, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_4)

        self.verticalLayout.setStretch(2, 10)

        self.gridLayout.addWidget(self.groupBox, 0, 0, 2, 1)

        self.widget = QWidget(Gragh_show)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 0)
        self.label_37 = QLabel(self.widget)
        self.label_37.setObjectName(u"label_37")
        self.label_37.setLayoutDirection(Qt.LeftToRight)
        self.label_37.setTextFormat(Qt.AutoText)
        self.label_37.setScaledContents(False)

        self.horizontalLayout.addWidget(self.label_37)

        self.statues_label = QLabel(self.widget)
        self.statues_label.setObjectName(u"statues_label")

        self.horizontalLayout.addWidget(self.statues_label)

        self.Clear_Button = QPushButton(self.widget)
        self.Clear_Button.setObjectName(u"Clear_Button")
        self.Clear_Button.setMaximumSize(QSize(100, 32))

        self.horizontalLayout.addWidget(self.Clear_Button)

        self.Collectbegin_Button = QPushButton(self.widget)
        self.Collectbegin_Button.setObjectName(u"Collectbegin_Button")
        self.Collectbegin_Button.setMaximumSize(QSize(100, 32))

        self.horizontalLayout.addWidget(self.Collectbegin_Button)

        self.Clearroom_Button = QPushButton(self.widget)
        self.Clearroom_Button.setObjectName(u"Clearroom_Button")
        self.Clearroom_Button.setMaximumSize(QSize(100, 32))

        self.horizontalLayout.addWidget(self.Clearroom_Button)

        self.Stop_Button = QPushButton(self.widget)
        self.Stop_Button.setObjectName(u"Stop_Button")
        self.Stop_Button.setMaximumSize(QSize(100, 32))

        self.horizontalLayout.addWidget(self.Stop_Button)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 10)

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
        self.groupBox_2.setTitle(QCoreApplication.translate("Gragh_show", u"\u4fdd\u5b58\u8def\u5f84", None))
        self.Folder_Button.setText(QCoreApplication.translate("Gragh_show", u"...", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Gragh_show", u"\u65f6\u957f\u8bbe\u7f6e", None))
        self.label_2.setText(QCoreApplication.translate("Gragh_show", u"\u6d17\u6c14:", None))
        self.label_34.setText(QCoreApplication.translate("Gragh_show", u"s", None))
        self.label_4.setText(QCoreApplication.translate("Gragh_show", u"\u91c7\u6837:", None))
        self.label_35.setText(QCoreApplication.translate("Gragh_show", u"s", None))
        self.label_9.setText(QCoreApplication.translate("Gragh_show", u"\u57fa\u7ebf:", None))
        self.label_36.setText(QCoreApplication.translate("Gragh_show", u"s", None))
        self.label_14.setText(QCoreApplication.translate("Gragh_show", u"\u5de5\u4f5c:", None))
        self.label_38.setText(QCoreApplication.translate("Gragh_show", u"s", None))
        self.Auto_Button.setText(QCoreApplication.translate("Gragh_show", u"\u81ea\u52a8\u6a21\u5f0f", None))
        self.InitPos_Button.setText(QCoreApplication.translate("Gragh_show", u"\u5f00\u59cb/\u521d\u59cb\u5316", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Gragh_show", u"\u4f20\u611f\u5668\u5217\u8868", None))
        self.label_37.setText(QCoreApplication.translate("Gragh_show", u"\u5f53\u524d\u72b6\u6001:", None))
        self.statues_label.setText("")
        self.Clear_Button.setText(QCoreApplication.translate("Gragh_show", u"\u57fa\u7ebf\u8c03\u6574", None))
        self.Collectbegin_Button.setText(QCoreApplication.translate("Gragh_show", u"\u5f00\u59cb\u91c7\u96c6", None))
        self.Clearroom_Button.setText(QCoreApplication.translate("Gragh_show", u"\u6e05\u6d17\u6c14\u5ba4", None))
        self.Stop_Button.setText(QCoreApplication.translate("Gragh_show", u"\u4e00\u952e\u6682\u505c", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Line_Show), QCoreApplication.translate("Gragh_show", u"\u66f2\u7ebf\u56fe", None))
    # retranslateUi

