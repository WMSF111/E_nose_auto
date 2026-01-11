''' 实验界面的绘图工作'''

import sys, os
# import requests_cache
from PySide6.QtGui import (
    QTextCursor
)

from PySide6.QtCore import (
    QObject,
    Signal,
)
from PySide6.QtGui import (
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
)
import pyqtgraph as pg
import Enose.global_var as g_var
import Enose.tool.serial_thread as sethread
from Enose.resource_ui.ui_pfile.Gragh_show import Ui_Gragh_show

Cleartime = 0

class MySignals(QObject):
    # 定义一种信号，两个参数 类型分别是： QTextBrowser 和 字符串
    # 调用 emit方法 发信号时，传入参数 必须是这里指定的 参数类型
    Draw = Signal(int)

    _serialComboBoxResetItems = Signal(list)
    _serialComboBoxClear = Signal()
    _setButtonText = Signal(str)
    _lineClear = Signal()

ms = MySignals()

class Action():
    def __init__(self, ui: QWidget) -> None:
        self = ui

    def Draw(self, receive_data):
        self.tb.insertPlainText(receive_data)

        # 获取到text光标,确保下次插入到内容最后
        textCursor = self.tb.textCursor()
        # 滚动到底部
        textCursor.movePosition(QTextCursor.End)
        # 设置光标到text中去
        self.tb.setTextCursor(textCursor)

        # 确保光标可见
        self.tb.ensureCursorVisible()


class Gragh_show_Init(QWidget, Ui_Gragh_show):
    def __init__(self):

        super(Gragh_show_Init, self).__init__()
        self.setupUi(self)
        # self.Data_Init() # 数据初始化
        # 初始化按键任务
        # self.Button_Init()
        # # 发送窗口放入字符
        # self.file_dir = os.getcwd() + "//log.txt"
        #
        # # 创建列表视图
        # self.model = QStandardItemModel()  # 标准表格模式
        # self.model.setHorizontalHeaderLabels(["传感器", "数据"])  # 设置表头
        # self.model.itemChanged.connect(self.check_check_state)  # 连接项目更改信号
        # self.Senser_stableView.setModel(self.model)  # 设置列表视图的模型

        # # 创建绘图窗口
        # self.plot1, self.curve, self.curve1 = self.set_graph_ui(60)
        # self.a = Action(self)
        # self.initMS()
        # self.ports = []
        # self.set_initial_baud_rate(g_var.selectBund)
        # self.Com_Dict = {}
        # self.ser = mythread.myserial()
        # self.initSerial()

    def Data_Init(self):
        # 创建保存接收字符的str
        self.receive_str = ''
        # 创建用于数据提取的缓存文件
        self.data_decoding_buffer = ''
        # tab标签标志
        self.tab_status = 1
        # 创建串口状态位
        self.serial_status = 0
        # 创建串口对象
        self.ser = sethread.myserial()
        # 横坐标长度
        self.historyLength = 300

        self.Cleartime = 0


    def Button_Init(self):
        ms.Draw.connect(self.a.Draw())

        # 设置tab上的关闭按钮是否显示
        self.tabWidget.setTabsClosable(False)
        # 数据清洗按钮
        self.Dataclear_Button.clicked.connect(self.Dataclear_Begin)
        # 开始采集按钮
        self.Collectbegin_Button.clicked.connect(self.a.Collect_begin)

        # 暂停采集按钮
        self.Pause_Button.clicked.connect(self.a.Collect_Pause)

        # 取消采集按钮
        self.Cancel_Button.clicked.connect(self.a.Collect_Cancel)

    def Dataclear_Begin(self):
        self.Cleartime = self.Cleartime_spinBox.currentText()
        self.openPort()

    def openPort(self): # 打开串口
        if self.ser.read_flag: # 如果串口存在
            self.ser.stop() # 关闭串口
        else:
            self.ser.setSer(g_var.Port_select, g_var.Bund_select) # 设置串口及波特率
            d = self.ser.open(ms.Draw.emit) # 打开串口，成功返回0，失败返回1， + str信息
            ms.Draw.emit(d[1])

    def set_graph_ui(self, x_max):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        # pg全局变量设置函数，antialias=True开启曲线抗锯齿
        pg.setConfigOptions(antialias=True)
        # 创建窗体，可实现数据界面布局自动管理
        win = pg.GraphicsLayoutWidget(show=True)

        # pg绘图窗口可以作为一个widget添加到GUI中的graph_layout，当然也可以添加到Qt其他所有的容器中
        self.Linegragh_Layout.addWidget(win)

        # 往窗口中增加一个图形命名为plot
        plot1 = win.addPlot()
        # 栅格设置函数
        plot1.showGrid(x=True, y=True)
		# 设置显示的范围
        plot1.setRange(xRange=[-5, x_max + 5], yRange=[0, self.historyLength], padding=0)
		# 设置标注的位置及内容
		# plot1.setLabel(axis='left', text='y / V', color='#ffff00')
        plot1.setLabel(axis='left')
		# plot1.setLabel(axis='bottom', text='x / point', units='s')
        plot1.setLabel(axis='bottom', text='x / point')
		# False代表线性坐标轴，True代表对数坐标轴
        plot1.setLogMode(x=False, y=False)

		# 也可以返回plot1对象，然后使用plot.plot(x,y,pen='r')来绘制曲线，x,y对应各轴数据
		# 需要去掉全部的curve，换成以上函数
        curve = plot1.plot()
        curve1 = plot1.plot()


        return plot1, curve, curve1


    def check_check_state(self, i):
        """
            检查并处理可复选项 i 的状态变化。

            参数:
                i: 一个可复选项对象，通常是一个表格单元格或列表项。
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Gragh_show_Init()
    window.show()
    sys.exit(app.exec())