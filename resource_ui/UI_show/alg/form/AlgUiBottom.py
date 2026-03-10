# -*- coding: utf-8 -*-

class BottomFrameManager:
    def __init__(self, ui_instance):
        self.ui = ui_instance

    def update_status(self, message):
        """更新状态栏消息"""
        self.ui.statusBar.setText(message)

    def update_info(self, message):
        """更新信息显示标签"""
        self.ui.label_info.setText(message)