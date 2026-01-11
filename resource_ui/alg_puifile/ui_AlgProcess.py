# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'AlgProcesssqLMca.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar,
    QPushButton, QRadioButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(900, 595)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(9)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.cBox_AlgGroup = QComboBox(self.groupBox_2)
        self.cBox_AlgGroup.setObjectName(u"cBox_AlgGroup")
        self.cBox_AlgGroup.setMinimumSize(QSize(120, 0))

        self.horizontalLayout_2.addWidget(self.cBox_AlgGroup)

        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")
        self.label.setStyleSheet(u"color: rgb(162, 162, 162);")

        self.horizontalLayout_2.addWidget(self.label)

        self.cBox_AlgName = QComboBox(self.groupBox_2)
        self.cBox_AlgName.setObjectName(u"cBox_AlgName")
        self.cBox_AlgName.setMinimumSize(QSize(120, 0))

        self.horizontalLayout_2.addWidget(self.cBox_AlgName)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)


        self.verticalLayout_4.addLayout(self.horizontalLayout_2)

        self.groupBox_5 = QGroupBox(self.groupBox_2)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.horizontalLayout_4 = QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")

        self.horizontalLayout_4.addLayout(self.formLayout)

        self.horizontalSpacer_4 = QSpacerItem(492, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_4)


        self.verticalLayout_4.addWidget(self.groupBox_5)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(588, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.rbtn_smallMin = QRadioButton(self.groupBox_2)
        self.rbtn_smallMin.setObjectName(u"rbtn_smallMin")

        self.horizontalLayout.addWidget(self.rbtn_smallMin)

        self.rbtn_small = QRadioButton(self.groupBox_2)
        self.rbtn_small.setObjectName(u"rbtn_small")

        self.horizontalLayout.addWidget(self.rbtn_small)

        self.rbtn_middle = QRadioButton(self.groupBox_2)
        self.rbtn_middle.setObjectName(u"rbtn_middle")
        self.rbtn_middle.setChecked(True)

        self.horizontalLayout.addWidget(self.rbtn_middle)

        self.rbtn_larger = QRadioButton(self.groupBox_2)
        self.rbtn_larger.setObjectName(u"rbtn_larger")

        self.horizontalLayout.addWidget(self.rbtn_larger)

        self.rbtn_largerMax = QRadioButton(self.groupBox_2)
        self.rbtn_largerMax.setObjectName(u"rbtn_largerMax")

        self.horizontalLayout.addWidget(self.rbtn_largerMax)

        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(20, 0))

        self.horizontalLayout.addWidget(self.label_3)

        self.btn_run = QPushButton(self.groupBox_2)
        self.btn_run.setObjectName(u"btn_run")
        font = QFont()
        font.setPointSize(11)
        font.setBold(False)
        self.btn_run.setFont(font)

        self.horizontalLayout.addWidget(self.btn_run)


        self.verticalLayout_4.addLayout(self.horizontalLayout)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.progressBar = QProgressBar(Form)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setMinimumSize(QSize(0, 20))
        self.progressBar.setMaximumSize(QSize(16777215, 20))
        self.progressBar.setMaximum(0)
        self.progressBar.setValue(-1)
        self.progressBar.setTextVisible(False)

        self.verticalLayout.addWidget(self.progressBar)

        self.groupBox_3 = QGroupBox(Form)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.plainTextEdit = QPlainTextEdit(self.groupBox_3)
        self.plainTextEdit.setObjectName(u"plainTextEdit")

        self.verticalLayout_3.addWidget(self.plainTextEdit)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)

        self.btn_scree_plot = QPushButton(self.groupBox_3)
        self.btn_scree_plot.setObjectName(u"btn_scree_plot")

        self.horizontalLayout_3.addWidget(self.btn_scree_plot)

        self.btn_scatter_plot = QPushButton(self.groupBox_3)
        self.btn_scatter_plot.setObjectName(u"btn_scatter_plot")

        self.horizontalLayout_3.addWidget(self.btn_scatter_plot)

        self.btn_source_data = QPushButton(self.groupBox_3)
        self.btn_source_data.setObjectName(u"btn_source_data")

        self.horizontalLayout_3.addWidget(self.btn_source_data)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)


        self.verticalLayout_3.addLayout(self.horizontalLayout_3)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.verticalSpacer = QSpacerItem(20, 95, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"\u7b97\u6cd5\u9009\u62e9", None))
        self.label.setText(QCoreApplication.translate("Form", u">>", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Form", u"\u7b97\u6cd5\u53c2\u6570\u8bbe\u7f6e", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u51fa\u56fe\u5c3a\u5bf8:", None))
        self.rbtn_smallMin.setText(QCoreApplication.translate("Form", u"\u5c0f", None))
        self.rbtn_small.setText(QCoreApplication.translate("Form", u"\u8f83\u5c0f", None))
        self.rbtn_middle.setText(QCoreApplication.translate("Form", u"\u9002\u4e2d", None))
        self.rbtn_larger.setText(QCoreApplication.translate("Form", u"\u8f83\u5927", None))
        self.rbtn_largerMax.setText(QCoreApplication.translate("Form", u"\u5927", None))
        self.label_3.setText("")
        self.btn_run.setText(QCoreApplication.translate("Form", u"\u8ba1\u7b97", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Form", u"\u8ba1\u7b97\u7ed3\u679c\u6982\u8ff0", None))
        self.plainTextEdit.setPlainText("")
        self.btn_scree_plot.setText(QCoreApplication.translate("Form", u"\u663e\u793a\u788e\u77f3\u56fe", None))
        self.btn_scatter_plot.setText(QCoreApplication.translate("Form", u"\u663e\u793a\u6563\u70b9\u56fe", None))
        self.btn_source_data.setText(QCoreApplication.translate("Form", u"\u663e\u793a\u539f\u6570\u636e", None))
    # retranslateUi

