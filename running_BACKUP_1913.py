import global_var
import os
import sys
import threading
import signal
import resource_ui.web_app
from resource_ui.modules import *
from resource_ui.web_app import run
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
import tool.UI_show.serial_show as se
from tool.UI_show.Gragn_show_ui import GraphShowWindow
from tool.UI_show.alg.form.AlgUi import AlgUIFrame # 第三版界面

# 高DPI支持 - 确保在高分辨率显示器上正确显示
#os.environ["QT_FONT_DPI"] = "150"

# 全局变量替换为模块级常量
WIDGETS = None  # UI组件全局引用，用于外部访问
APP_TITLE = "实验平台"  # 应用程序标题
APP_DESCRIPTION = "智能电子鼻实验及算法平台"  # 应用程序描述


class MainWindow(QMainWindow):
    """主窗口类，负责应用程序的UI和功能管理"""

    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self._init_ui()
        self._setup_window()
        self._init_modules()
        # 连接信号和槽
        self._connect_signals()
        self._apply_theme()
        self.showMaximized() #.showNormal()  # 可以切换为 showMaximized() 以全屏显示
        self._set_initial_page()

    def _init_ui(self):
        """初始化UI组件"""
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        UIFunctions.uiDefinitions(self)
        # 设置全局UI引用，便于其他模块访问
        global WIDGETS
        WIDGETS = self.ui

    def _setup_window(self):
        """设置窗口基本属性"""
        # 设置窗口标题
        self.setWindowTitle(APP_TITLE)

        WIDGETS.titleLeftApp.setText(APP_TITLE)
        WIDGETS.titleRightInfo.setText(APP_DESCRIPTION)

        # 启用自定义标题栏（Mac/Linux可能需要设为False）
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

    def _init_modules(self):
        """初始化所有功能模块"""
        # 串口设置
        self.serial_init = se.Serial_Init()
        WIDGETS.stackedWidget.addWidget(self.serial_init)

        # 测试阶段
        self.test_show = GraphShowWindow()
        WIDGETS.stackedWidget.addWidget(self.test_show)

        # 算法选择
        self.alg_show = AlgUIFrame() #AlgInit()#AlgUiFrame()#AlgUIFrame()
        WIDGETS.stackedWidget.addWidget(self.alg_show)

        # Flask线程变量，用于启动Web应用
        self.flask_thread = None

    def _connect_signals(self):
        """连接所有信号和槽函数"""
        WIDGETS.toggleButton.clicked.connect(
            lambda: UIFunctions.toggleMenu(self, True)
        )

        WIDGETS.btn_serial.clicked.connect(self._on_button_click)  # 串口页面
        WIDGETS.btn_test.clicked.connect(self._on_button_click)  # 测试页面
        WIDGETS.btn_alg.clicked.connect(self._on_button_click)  # 算法页面
        WIDGETS.btn_ai.clicked.connect(self._on_button_click)  # AI/Web页面

    def _apply_theme(self):
        """应用主题样式到应用程序"""
        use_custom_theme = True

        if use_custom_theme:
            theme_file = global_var.themeFile
            UIFunctions.theme(self, theme_file, True)
            AppFunctions.setThemeHack(self)

    def _set_initial_page(self):
        """设置应用程序启动时的默认页面"""
        WIDGETS.stackedWidget.setCurrentWidget(self.serial_init)
        WIDGETS.btn_serial.setStyleSheet(UIFunctions.selectMenu(WIDGETS.btn_serial.styleSheet()))

    def _on_button_click(self):
        """处理所有按钮的点击事件"""
        button = self.sender()
        button_name = button.objectName()

        button_handlers = {
            'btn_serial': self._handle_serial_button,  # 串口设置
            'btn_test': self._handle_test_button,  # 测试阶段
            'btn_alg': self._handle_algorithm_button,  # 算法选择
            'btn_ai': self._handle_webapp_button,  # 大模型
        }

        handler = button_handlers.get(button_name)
        if handler:
            handler(button)

    def _handle_serial_button(self, button):
        if self.test_show.ser:
            self.test_show.SerStop()
        """处理串口按钮点击 - 切换到串口页面"""
        WIDGETS.stackedWidget.setCurrentWidget(self.serial_init)
        UIFunctions.resetStyle(self, 'btn_serial')
        button.setStyleSheet(UIFunctions.selectMenu(button.styleSheet()))

    def _handle_test_button(self, button):
        """处理测试按钮点击 - 切换到图表显示页面"""
        self._stop_serial_connection()
        self.test_show = GraphShowWindow()
        WIDGETS.stackedWidget.addWidget(self.test_show)  # 将串口界面添加到 stackedWidget
        WIDGETS.stackedWidget.setCurrentWidget(self.test_show)
        UIFunctions.resetStyle(self, 'btn_test')
        button.setStyleSheet(UIFunctions.selectMenu(button.styleSheet()))

    def _handle_algorithm_button(self, button):
        """处理算法按钮点击 - 切换到算法配置页面"""
        WIDGETS.stackedWidget.setCurrentWidget(self.alg_show)
        UIFunctions.resetStyle(self, 'btn_alg')
        button.setStyleSheet(UIFunctions.selectMenu(button.styleSheet()))

    def _handle_webapp_button(self, button):
        """处理Web应用按钮点击 - 启动Flask Web服务器"""
        try:
            self.flask_thread = threading.Thread(target=run, daemon=True)
            self.flask_thread.start()
            resource_ui.web_app.open_browser()
            UIFunctions.resetStyle(self, 'btn_ai')
            button.setStyleSheet(UIFunctions.selectMenu(button.styleSheet()))
        except Exception:
            self._show_error_message("无法打开链接:")

    def _stop_serial_connection(self):
        """停止串口连接，确保资源正确释放"""
        try:
            if global_var.Port_select == "":
                self._show_error_message("请先选择串口")
            elif self.serial_init.ser:
                if self.serial_init.ser.read_flag:  # 打开状态则关闭串口
                    self.serial_init.ser.stop()  # 关闭串口
                    if global_var.Auto_falg == True:
                        self.serial_init.ser1.stop()  # 关闭串口
                    se.ms._setButtonText.emit("断开状态")
                    self.serial_init.ser_open_look_ui(True)
        except:
            self._show_error_message("请先选择串口")

    def _show_error_message(self, text):
        """显示错误消息对话框"""
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Icon.Critical)
        message_box.setWindowTitle("错误")
        message_box.setText(text)
        message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        message_box.exec()

    def resizeEvent(self, event):
        """窗口大小调整事件处理"""
        super().resizeEvent(event)
        UIFunctions.resize_grips(self)

    def mousePressEvent(self, event):
        """鼠标按下事件处理"""
        super().mousePressEvent(event)
        self.dragPos = event.globalPos()

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        print("关闭事件")
        self._close_modules(event)
        self._graceful_shutdown()
        event.accept()

    def _close_modules(self, event):
        """关闭所有功能模块，确保资源正确释放"""
        modules_to_close = [
            self.serial_init,
            self.test_show,
            self.alg_show,
        ]

        for module in modules_to_close:
            if module and hasattr(module, 'closeEvent'):
                try:
                    module.closeEvent(event)
                except Exception as e:
                    print(f"关闭模块时出错: {e}")

    def _graceful_shutdown(self):
        """关闭应用程序，先尝试正常关闭，失败则强制结束"""
        try:
            if hasattr(resource_ui.web_app, 'shutdown'):
                resource_ui.web_app.shutdown()
        except Exception:
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)


def main():
    """应用程序主函数"""
    app = QApplication(sys.argv)
<<<<<<< HEAD
    # 步骤1：设置Fusion样式（核心，避免系统原生样式干扰）
    app.setStyle("Fusion")

    # 步骤2：构建浅色主题的固定调色板
    light_palette = QPalette()
    # 核心配色：模拟系统浅色主题默认值
    light_palette.setColor(QPalette.Window, QColor(240, 240, 240))  # 窗口背景（浅灰）
    light_palette.setColor(QPalette.WindowText, QColor(0, 0, 0))  # 窗口文字（黑色）
    light_palette.setColor(QPalette.Base, QColor(255, 255, 255))  # 输入框/编辑区背景（白色）
    light_palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))  # 交替行背景（浅灰）
    light_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))  # 提示框背景（白色）
    light_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))  # 提示框文字（黑色）
    light_palette.setColor(QPalette.Text, QColor(0, 0, 0))  # 普通文字（黑色）
    light_palette.setColor(QPalette.Button, QColor(236, 248, 248))  # 按钮背景（浅蓝）
    light_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))  # 按钮文字（黑色）
    light_palette.setColor(QPalette.BrightText, QColor(255, 255, 255))  # 高亮文字（白色）
    light_palette.setColor(QPalette.Highlight, QColor(66, 133, 244))  # 选中高亮（谷歌蓝，系统浅色默认）
    light_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # 高亮文字（白色）

    # 步骤3：全局应用浅色调色板
    app.setPalette(light_palette)

    # 步骤4：强制设置color-scheme为light（Qt 5.15+支持，强化浅色优先级）
    app.setStyleSheet("QApplication { color-scheme: light; }")
=======
>>>>>>> 954cd1191aee3c61d370d5aaec6cf1a0f2a72684
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    return app.exec()


if __name__ == "__main__":
    # 程序入口
    sys.exit(main())