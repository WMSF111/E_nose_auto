import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
import global_var as g_var
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget, QFileDialog, QApplication
from PySide6.QtGui import QIcon, QTextCursor
import tool.serial_thread as mythread
if g_var.Auto_falg == True:
    from resource_ui.ui_pfile.Serial import Ui_Serial
else:
    from resource_ui.ui_pfile.Serial_1 import Ui_Serial


class MySignals(QObject):
    # 定义一种信号，两个参数 类型分别是： QTextBrowser 和 字符串
    # 调用 emit方法 发信号时，传入参数 必须是这里指定的 参数类型
    print = Signal(str)

    _serialComboBoxResetItems = Signal(list)
    _serialComboBoxClear = Signal()
    _setButtonText = Signal(str)
    _lineClear = Signal()


ms = MySignals()


class Action():
    def __init__(self, ui: QWidget) -> None:
        self.ui = ui

    def _serialComboBoxResetItems(self, texts: list):
        self.ui.serialComboBox.clear()
        self.ui.serialComboBox.addItems(texts)
        if g_var.Auto_falg == True:
            self.ui.serialComboBox2.clear()
            self.ui.serialComboBox2.addItems(texts)

    def _serialComboBoxclear(self):
        self.ui.serialComboBox.clear()
        if g_var.Auto_falg == True:
            self.ui.serialComboBox2.clear()

    def _setButtonText(self, text: str):
        self.ui.connectButton.setText(text)

    def _lineClear(self):
        self.ui.sendEdit.clear()

    def print(self, receive_data):
        self.ui.tb.insertPlainText(receive_data)

        # 获取到text光标,确保下次插入到内容最后
        textCursor = self.ui.tb.textCursor()
        # 滚动到底部
        textCursor.movePosition(QTextCursor.End)
        # 设置光标到text中去
        self.ui.tb.setTextCursor(textCursor)

        # 确保光标可见
        self.ui.tb.ensureCursorVisible()



class Serial_Init(QWidget):
    def __init__(self):

        super(Serial_Init, self).__init__()
        self.ui = Ui_Serial()
        self.ui.setupUi(self)

        self.a = Action(self.ui)
        self.initMS()
        self.ports = []
        self.Port_select = ""
        self.Com_Dict = {}

    def set_initial_baud_rate(self, baud_rate):
        """
        设置初始波特率。
        :param baud_rate: 初始波特率值
        """
        # 遍历 QComboBox 的所有选项
        for i in range(self.ui.baudComboBox.count()):
            # 获取当前选项的文本（波特率值）
            current_rate = int(self.ui.baudComboBox.itemText(i))
            # 如果当前选项的波特率值等于目标波特率，则设置为当前选中项
            if current_rate == baud_rate:
                self.ui.baudComboBox.setCurrentIndex(i)
                break

    def initMS(self):
        ms._serialComboBoxResetItems.connect(self.a._serialComboBoxResetItems)
        ms._serialComboBoxClear.connect(self.a._serialComboBoxclear)
        ms._setButtonText.connect(self.a._setButtonText)
        ms.print.connect(self.a.print)
        ms._lineClear.connect(self.a._lineClear)
        self.ui.CheckButton.clicked.connect(self.initallSerial)  # 检查按钮选择
        self.ui.connectButton.clicked.connect(self.openPort)  # 链接按钮选择
        self.ui.sendButton.clicked.connect(self.send)  # sendButton联系
        # 串口信息显示
        self.ui.serialComboBox.setEnabled(False)
        self.ui.serialComboBox.currentTextChanged.connect(
            lambda: self.initComboBox(self.ui.serialComboBox, self.ui.statues)
        )
        if g_var.Auto_falg == True:
            self.ui.serialComboBox2.setEnabled(False)
            self.ui.serialComboBox2.currentTextChanged.connect(
                lambda: self.initComboBox(self.ui.serialComboBox2, self.ui.statues_3)
            )
        self.ui.saveButton.clicked.connect(self.savefile)  # 保存文件按钮
        self.ui.clearButton.clicked.connect(self.ui.tb.clear)  # 清除按钮

    def initComboBox(self, serialComboBox, statues): # 初始化下拉列表
        self.port_imf(serialComboBox, statues)  # 显示串口信息
        ports, self.Com_Dict = mythread.getPortList()  # 获取串口列表
        if self.ports != ports:  # 如果串口不是所选的
            self.ports = ports
            if g_var.Port_select not in [i.name for i in self.ports]:
                self.ser.read_flag = False
                print([i.name for i in self.ports], g_var.Port_select)
                print("初始化列表失败")
            ms._serialComboBoxResetItems.emit([i.name for i in self.ports])  # 添加所有串口

    def initallSerial(self):
        self.ui.serialComboBox.setEnabled(True)
        if g_var.Auto_falg == True:
            self.ui.serialComboBox2.setEnabled(True)
            sconfig = [" ", 115200, " ", 9600]
            self.smng = mythread.SerialsMng(sconfig)
            self.ser = self.smng.ser_arr[0]
            self.ser1 = self.smng.ser_arr[1]
            self.initComboBox(self.ui.serialComboBox, self.ui.statues)  # 初始化ser列表
            self.initComboBox(self.ui.serialComboBox2, self.ui.statues_3) # 初始化ser1列表
        else:
            sconfig = [" ", 115200,]
            self.smng = mythread.SerialsMng(sconfig)
            self.ser = self.smng.ser_arr[0]
            self.initComboBox(self.ui.serialComboBox, self.ui.statues)  # 初始化ser列表

    def openPort(self):  # 打开串口
        print("read_flag:", str(self.ser.read_flag))
        if self.ser.read_flag: # 打开状态则关闭串口
            self.ser.stop()  # 关闭串口
            if g_var.Auto_falg == True:
                self.ser1.stop()  # 关闭串口
            ms._setButtonText.emit("断开状态")
            self.ser_open_look_ui(True)
        else: # 关闭状态则连接串口
            # 先重设串口设置
            g_var.Port_select = self.ui.serialComboBox.currentText()  # 串口选择
            self.ser.setSer(g_var.Port_select, g_var.Bund_select)  # 设置串口及波特率 重设
            if g_var.Auto_falg == True:
                g_var.Port_select2 = self.ui.serialComboBox2.currentText()  # 串口选择
                self.ser1.setSer(g_var.Port_select2, g_var.Bund_select2)  # 设置串口及波特率 重设
            #再打开串口
            d = self.ser.open(ms.print.emit)  # 打开串口，成功返回0，失败返回1， + str信息
            print(d)
            ms.print.emit(d[1])
            if g_var.Auto_falg == True:
                d = self.ser1.open(ms.print.emit, flag = 1)
                print(d)
                ms.print.emit(d[1])
            if d[0] == 0:
                ms._setButtonText.emit("连接状态")
                self.ser_open_look_ui(False)


    def send(self):
        text = self.ui.sendEdit.text()
        if not (text == "") or not (text is None):
            if self.ser.read_flag:
                if text[0:2] == "55":
                    if g_var.Auto_falg == True:
                        self.ser1.write(text)
                        ms.print.emit(text + '\n')
                else:
                    self.ser.write(text)
                    ms.print.emit(text)
                ms._lineClear.emit()

    def showAbout(self):
        self.ui.msg.show()

    # 串口信息
    def port_imf(self, serialComboBox, statues):
        # 显示选定的串口的详细信息
        imf_s = serialComboBox.currentText()
        if imf_s != "":
            statues.setText(self.Com_Dict[serialComboBox.currentText()])
        else:
            statues.setText("     无串口信息")

    def savefile(self):
        filename = QFileDialog.getSaveFileName(None, "open file", "/", "TEXT Files(*.txt)")
        # print(filename)
        if filename[0] == "" or filename is None:
            return
        try:
            with open(filename[0], "w") as f:
                text = self.ui.tb.toPlainText()
                f.write(text)
            ms.print.emit("保存到" + filename[0])
        except Exception as e:
            ms.print.emit("保存失败" + str(e))

        # 锁定和解锁ui 无法点击串口信息及波特率

    def ser_open_look_ui(self, status): #禁止读取列表
        self.ui.serialComboBox.setEnabled(status)
        if g_var.Auto_falg == True:
            self.ui.serialComboBox2.setEnabled(status)

    def closeEvent(self, event = None):
        """点击右上角 X 时调用"""
        # 1. 停止串口线程
        if hasattr(self, 'ser') and self.ser.read_flag:
            self.ser.stop()
        if g_var.Auto_falg == True:
            if hasattr(self, 'ser1') and self.ser1.read_flag:
                self.ser1.stop()
        if event != None:
            event.accept()  # 允许窗口真正关闭


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     app.setStyle("WindowsVista")  # 强制使用 WindowsVista 主题
#     window = Serial_Init()
#     window.show()
#     sys.exit(app.exec())