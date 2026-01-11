# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

# MAIN FILE
# ///////////////////////////////////////////////////////////////
from running import *

# WITH ACCESS TO MAIN WINDOW WIDGETS
# ///////////////////////////////////////////////////////////////
class AppFunctions(MainWindow):
    def setThemeHack(self):
        # 设置按钮左侧和右侧的蓝色背景
        Settings.BTN_LEFT_BOX_COLOR = "background-color: #215b5d;"  # 蓝色背景
        Settings.BTN_RIGHT_BOX_COLOR = "background-color: #215b5d;"  # 蓝色背景

        # 更新菜单选中样式为蓝色渐变
        Settings.MENU_SELECTED_STYLESHEET = MENU_SELECTED_STYLESHEET = """
        border-left: 22px solid qlineargradient(spread:pad, x1:0.034, y1:0, x2:0.216, y2:0, stop:0.499 rgba(85, 170, 255, 255),
         stop:0.5 rgba(0, 0, 255, 0));
         background-color: #215b5d;  /* 更深的蓝色背景 */
        """

