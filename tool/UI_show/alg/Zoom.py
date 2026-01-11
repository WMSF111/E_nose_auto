
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QLabel)
from PySide6.QtGui import QPixmap, QWheelEvent


class ZoomableLabel(QLabel):
    """支持鼠标滚轮缩放图片的QLabel"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.current_scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.scale_step = 0.1

    def setPixmap(self, pixmap):
        """设置图片"""
        self.original_pixmap = pixmap
        self.current_scale = 1.0
        if pixmap:
            scaled_pixmap = pixmap.scaled(
                pixmap.size() * self.current_scale,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)
        else:
            super().setPixmap(QPixmap())

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件 - 缩放图片"""
        if self.original_pixmap is None:
            return

        # 获取滚轮滚动角度
        delta = event.angleDelta().y()

        # 计算新缩放比例
        if delta > 0:
            self.current_scale = min(self.current_scale + self.scale_step, self.max_scale)
        else:
            self.current_scale = max(self.current_scale - self.scale_step, self.min_scale)

        # 缩放图片
        scaled_pixmap = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.current_scale,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)



    def reset_zoom(self):
        """重置缩放"""
        if self.original_pixmap:
            self.current_scale = 1.0
            super().setPixmap(self.original_pixmap)

    def save_image(self, default_path=None):
        """保存图片"""
        if self.original_pixmap is None:
            return False, "没有图片可保存"

        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片",
            default_path or "未命名.png",
            "PNG图片 (*.png);;JPEG图片 (*.jpg *.jpeg);;BMP图片 (*.bmp);;所有文件 (*.*)"
        )

        if file_path:
            success = self.original_pixmap.save(file_path)
            if success:
                return True, f"图片已保存到: {file_path}"
            else:
                return False, f"保存图片失败: {file_path}"
        return False, "取消保存"