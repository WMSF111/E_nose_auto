# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Predict.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget)

class Ui_Pre_show(object):
    def setupUi(self, Pre_show):
        if not Pre_show.objectName():
            Pre_show.setObjectName(u"Pre_show")
        Pre_show.resize(300, 238)
        self.verticalLayout = QVBoxLayout(Pre_show)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.pushButton = QPushButton(Pre_show)
        self.pushButton.setObjectName(u"pushButton")

        self.verticalLayout.addWidget(self.pushButton)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_3 = QLabel(Pre_show)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_2.addWidget(self.label_3)

        self.FilePath_lineEdit = QLineEdit(Pre_show)
        self.FilePath_lineEdit.setObjectName(u"FilePath_lineEdit")

        self.horizontalLayout_2.addWidget(self.FilePath_lineEdit)

        self.toolButton = QToolButton(Pre_show)
        self.toolButton.setObjectName(u"toolButton")

        self.horizontalLayout_2.addWidget(self.toolButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setLabelAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.formLayout.setFormAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label = QLabel(Pre_show)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label)

        self.Pre_ComboBox = QComboBox(Pre_show)
        self.Pre_ComboBox.setObjectName(u"Pre_ComboBox")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.Pre_ComboBox)

        self.label_2 = QLabel(Pre_show)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_2)

        self.Valchoose_ComboBox = QComboBox(Pre_show)
        self.Valchoose_ComboBox.setObjectName(u"Valchoose_ComboBox")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.Valchoose_ComboBox)


        self.verticalLayout.addLayout(self.formLayout)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 3)

        self.retranslateUi(Pre_show)

        QMetaObject.connectSlotsByName(Pre_show)
    # setupUi

    def retranslateUi(self, Pre_show):
        Pre_show.setWindowTitle(QCoreApplication.translate("Pre_show", u"Form", None))
        self.pushButton.setText(QCoreApplication.translate("Pre_show", u"\u786e\u5b9a", None))
        self.label_3.setText(QCoreApplication.translate("Pre_show", u"\u6587\u4ef6\u8def\u5f84", None))
        self.toolButton.setText(QCoreApplication.translate("Pre_show", u"...", None))
        self.label.setText(QCoreApplication.translate("Pre_show", u"\u6ee4\u6ce2\u51fd\u6570", None))
        self.label_2.setText(QCoreApplication.translate("Pre_show", u"\u503c\u9009\u62e9", None))
    # retranslateUi

