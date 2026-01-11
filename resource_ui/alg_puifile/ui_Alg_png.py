# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Alg_pngaxxIlo.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

class Ui_Alg_show(object):
    def setupUi(self, Alg_show):
        if not Alg_show.objectName():
            Alg_show.setObjectName(u"Alg_show")
        Alg_show.resize(1140, 710)
        self.verticalLayout = QVBoxLayout(Alg_show)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 10)
        self.label_png = QLabel(Alg_show)
        self.label_png.setObjectName(u"label_png")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_png.sizePolicy().hasHeightForWidth())
        self.label_png.setSizePolicy(sizePolicy)
        self.label_png.setScaledContents(False)
        self.label_png.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.label_png)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.btn_save_pic = QPushButton(Alg_show)
        self.btn_save_pic.setObjectName(u"btn_save_pic")

        self.horizontalLayout.addWidget(self.btn_save_pic)

        self.btn_save_data = QPushButton(Alg_show)
        self.btn_save_data.setObjectName(u"btn_save_data")

        self.horizontalLayout.addWidget(self.btn_save_data)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(Alg_show)

        QMetaObject.connectSlotsByName(Alg_show)
    # setupUi

    def retranslateUi(self, Alg_show):
        Alg_show.setWindowTitle(QCoreApplication.translate("Alg_show", u"Form", None))
        self.label_png.setText("")
        self.btn_save_pic.setText(QCoreApplication.translate("Alg_show", u" \u56fe\u7247\u53e6\u5b58\u4e3a...", None))
        self.btn_save_data.setText(QCoreApplication.translate("Alg_show", u"\u6570\u636e\u53e6\u5b58\u4e3a...", None))
    # retranslateUi

