'''测试算法页面'''

import sys
from PySide6.QtWidgets import QApplication, QWidget, QFileDialog
from resource_ui.alg_puifile.Alg_show import Ui_Alg_show  # 导入生成的 UI 类
import Enose.data_file.transfo as transfo
import matplotlib.pyplot as plt
import Enose.global_var as glov
from Enose.data_file.alg_tabadd import ALG_TAB_ADD

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class AlgShow_Init(QWidget, Ui_Alg_show):
    def __init__(self):
        super(AlgShow_Init, self).__init__()
        self.setupUi(self)  # 设置 UI 界面
        self.tabadd = ALG_TAB_ADD(self)
        self.ButInit()
        self.ComboInit()


    def ButInit(self):
        # 绑定按钮点击事件
        self.toolButton.clicked.connect(self.select_file)
        # 设置 TabWidget 的关闭按钮策略
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.tabCloseRequested.connect(self.remove_tab)

    def remove_tab(self, index):
        """删除指定的 Tab"""
        self.tabWidget.removeTab(index)

    def ComboInit(self):
        self.Di_Re_ComboBox.addItems(["无", "PCA", "LDA", "选项3"])  # 添加选项
        self.Di_Re_ComboBox.setCurrentIndex(0)  # 设置默认选择为第一个选项（"无"）
        self.Di_Re_ComboBox.currentIndexChanged.connect(self.tabadd.Di_Re_Combo_select)
        self.Classify_ComboBox.addItems(["无", "线性回归", "选项2", "选项3"])  # 添加选项
        self.Classify_ComboBox.setCurrentIndex(0)  # 设置默认选择为第一个选项（"无"）
        self.Classify_ComboBox.currentIndexChanged.connect(self.tabadd.Classify_Combo_select)
        self.Cluster_ComboBox.addItems(["无", "选项1", "选项2", "选项3"])  # 添加选项
        self.Cluster_ComboBox.setCurrentIndex(0)  # 设置默认选择为第一个选项（"无"）
        # self.Cluster_ComboBox.currentIndexChanged.connect(self.Cluster_Combo_select)
        self.Reg_ComboBox.addItems(["无", "选项1", "选项2", "选项3"])  # 添加选项
        self.Reg_ComboBox.setCurrentIndex(0)  # 设置默认选择为第一个选项（"无"）
        # self.Reg_ComboBox.currentIndexChanged.connect(self.Reg_Combo_select)

    def select_file(self):
        """弹出文件夹选择对话框"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",  # 对话框标题
            "",  # 默认打开的目录（空表示当前目录）
        )
        if folder_path:  # 如果用户选择了文件夹（而不是取消）
            self.FilePath_lineEdit.setText(folder_path)  # 显示到 QLineEdit
            glov.folder_path = folder_path
            transfo.UI_TXT_TO.unit_traintxt(glov.folder_path) # 遍历文件夹中的.txt合并成trainfile.txt
            textEdit_DataFrame = transfo.UI_TXT_TO.txt_to_dataframe(glov.trainfile_txt_path) # 读取trainfile.txt并显示到数据源看板
            self.DataBroad.append(textEdit_DataFrame.to_string(index=False))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = AlgShow_Init()
    main_window.show()
    sys.exit(app.exec())
