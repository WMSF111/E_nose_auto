from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from resource_ui.alg_puifile.ui_Alg_png import Ui_Alg_show
import shutil
import os
from pathlib import Path

class PngViewer(QDialog, Ui_Alg_show):
    """显示png图"""

    def __init__(self, png_path, work_path):
        super(PngViewer, self).__init__()
        self.setupUi(self)  # 设置 UI 界面
        self.png_path = png_path                # 图片路径
        self.work_path = work_path              # 工作目录
        # new_path = os.path.normpath(png_path)   # 路径符修改

        self.pixmap = QPixmap(png_path)
        self.label_png.setPixmap(self.pixmap)

        # 初始化缩放因子
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.zoom_step = 0.1

        # 安装事件过滤器来捕获鼠标滚轮事件
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # 按钮事件
        self.btn_save_pic.clicked.connect(self.save_png)    # 图片另存
        self.btn_save_data.clicked.connect(self.save_data)  # 图片数据另存

    def wheelEvent(self, event: QWheelEvent):
        """主窗口的滚轮事件"""
        # 检查是否在label上
        if self.label_png.underMouse():
            delta = event.angleDelta().y()
            if delta > 0:
                # 滚轮向上，放大
                self.zoom_in()
            else:
                # 滚轮向下，缩小
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def zoom_in(self):
        """放大"""
        self.scale_factor = min(self.scale_factor + self.zoom_step, self.max_scale)
        self.update_display()
        print(f"Zoom in: {self.scale_factor:.2f}")

    def zoom_out(self):
        """缩小"""
        self.scale_factor = max(self.scale_factor - self.zoom_step, self.min_scale)
        self.update_display()
        print(f"Zoom out: {self.scale_factor:.2f}")

    def update_display(self):
        """更新图片显示"""
        scaled_pixmap = self.pixmap.scaled(
            self.pixmap.size() * self.scale_factor,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label_png.setPixmap(scaled_pixmap)

    def save_png(self):
        """图片另存为"""
        dst_path, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口
            "保存图片",  # 对话框标题
            "",  # 初始目录（空表示当前目录）
            "png文件 (*.png);;所有文件 (*)"  # 文件过滤器
        )
        if dst_path:
            shutil.copy(self.png_path, dst_path)
            QMessageBox.information(None, "提示", f"图片已存储到：{dst_path}")

    def save_data(self):
        """图数据另存为"""
        src_path = os.path.join(self.work_path, "scree_plot.json")
        dst_path, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口
            "保存图数据",  # 对话框标题
            "",  # 初始目录（空表示当前目录）
            "csv文件 (*.csv);;所有文件 (*)"  # 文件过滤器
        )
        if dst_path:
            shutil.copy(src_path, dst_path)
            QMessageBox.information(None, "提示", f"图数据已存储到：{dst_path}")