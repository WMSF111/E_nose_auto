# -*- coding: utf-8 -*-

import sys
from PySide6.QtWidgets import QApplication, QWidget

# 导入各个模块
from resource_ui.alg_puifile.ui_Alg_Frame_New import Ui_Form
from tool.UI_show.alg.form.AlgUiLeft import LeftFrameManager
from tool.UI_show.alg.form.AlgUiBottom import BottomFrameManager
from tool.UI_show.alg.form.AlgUiRight import RightFrameManager


class AlgUIFrame(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 初始化各个管理器
        self.bottom_manager = BottomFrameManager(self)
        self.left_manager = LeftFrameManager(self)
        self.right_manager = RightFrameManager(self)

        # 绑定事件
        self.setup_connections()

    def setup_connections(self):
        """设置所有信号与槽的连接"""

        # 连接右侧按钮点击事件
        # self.btn_scree_plot.clicked.connect(lambda: self.right_manager.open_new_tab("碎石图"))
        # self.btn_scatter_plot.clicked.connect(lambda: self.right_manager.open_new_tab("散点图"))
        # self.btn_original_data.clicked.connect(lambda: self.right_manager.open_new_tab("原数据"))



    def get_selected_files(self):
        """获取所有选中的文件路径"""
        return self.left_manager.get_selected_files()

    def get_imported_folders(self):
        """获取所有导入的文件夹信息"""
        return self.left_manager.get_imported_folders()


# 测试调起窗口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AlgUIFrame()
    window.show()
    sys.exit(app.exec())