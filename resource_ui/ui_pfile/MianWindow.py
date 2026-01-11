# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MianWindoweDiZvh.ui'
##
## Created by: Qt User Interface Compiler version 6.7.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QHeaderView,
    QMainWindow, QSizePolicy, QStatusBar, QTreeWidget,
    QTreeWidgetItem, QWidget)
import icon_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1077, 706)
        self.Open_action = QAction(MainWindow)
        self.Open_action.setObjectName(u"Open_action")
        icon = QIcon()
        icon.addFile(u":/\u6587\u4ef6/\u6253\u5f00.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.Open_action.setIcon(icon)
        self.Newfile_action = QAction(MainWindow)
        self.Newfile_action.setObjectName(u"Newfile_action")
        icon1 = QIcon()
        icon1.addFile(u":/\u6587\u4ef6/\u65b0\u5efa.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.Newfile_action.setIcon(icon1)
        self.action = QAction(MainWindow)
        self.action.setObjectName(u"action")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setMinimumSize(QSize(1077, 630))
        self.centralwidget.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.horizontalLayout_3 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.treeWidget = QTreeWidget(self.centralwidget)
        QTreeWidgetItem(self.treeWidget)
        QTreeWidgetItem(self.treeWidget)
        QTreeWidgetItem(self.treeWidget)
        QTreeWidgetItem(self.treeWidget)
        self.treeWidget.setObjectName(u"treeWidget")

        self.horizontalLayout_3.addWidget(self.treeWidget)

        self.show_widget = QWidget(self.centralwidget)
        self.show_widget.setObjectName(u"show_widget")
        self.gridLayout_2 = QGridLayout(self.show_widget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.show_Layout = QGridLayout()
        self.show_Layout.setObjectName(u"show_Layout")

        self.gridLayout_2.addLayout(self.show_Layout, 0, 0, 1, 1)


        self.horizontalLayout_3.addWidget(self.show_widget)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.Open_action.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00", None))
        self.Newfile_action.setText(QCoreApplication.translate("MainWindow", u"\u65b0\u5efa", None))
        self.action.setText(QCoreApplication.translate("MainWindow", u"\u4f20\u611f\u5668\u8bbe\u7f6e", None))
        ___qtreewidgetitem = self.treeWidget.headerItem()
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("MainWindow", u"\u5b9e\u9a8c", None));

        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        ___qtreewidgetitem1 = self.treeWidget.topLevelItem(0)
        ___qtreewidgetitem1.setText(0, QCoreApplication.translate("MainWindow", u"\u4f20\u611f\u5668\u8bbe\u7f6e", None));
        ___qtreewidgetitem2 = self.treeWidget.topLevelItem(1)
        ___qtreewidgetitem2.setText(0, QCoreApplication.translate("MainWindow", u"\u6d4b\u8bd5\u9636\u6bb5", None));
        ___qtreewidgetitem3 = self.treeWidget.topLevelItem(2)
        ___qtreewidgetitem3.setText(0, QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u8ba1\u7b97", None));
        ___qtreewidgetitem4 = self.treeWidget.topLevelItem(3)
        ___qtreewidgetitem4.setText(0, QCoreApplication.translate("MainWindow", u"\u5927\u6a21\u578b\u5206\u6790", None));
        self.treeWidget.setSortingEnabled(__sortingEnabled)

    # retranslateUi

