import  sys, re, random, threading, logging
import time
from io import StringIO
from datetime import datetime
import os
from PySide6.QtCore import (
    Qt, QObject, Signal, QEvent, QTimer
)
from PySide6.QtWidgets import  QWidget, QHeaderView, QFileDialog, QApplication
from PySide6.QtGui import QColor, QStandardItemModel, QStandardItem, QBrush
import pandas as pd
import tool.serial_thread as mythread
import pyqtgraph as pg
import global_var as g_var
from itertools import cycle
import tool.Serial_opea as SO
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import resource_ui.alg_puifile.pic_tab_add as Tab_add
import tkinter as tk
from tkinter import messagebox
if g_var.Auto_falg == True:
    from resource_ui.ui_pfile.ChooseAndShow import Ui_Gragh_show
else:
    from resource_ui.ui_pfile.ChooseAndShow_1 import Ui_Gragh_show
class MySignals(QObject):
    _draw_open = Signal()
    _draw_close = Signal()
    _Clear_Button = Signal(bool)
    _Collectbegin_Button = Signal(bool)
    _Clearroom_Button = Signal(bool)
    _statues_label = Signal(str)
    _ClearDraw = Signal()
    _print = Signal(int)
    _attendtime_spinBox = Signal(int)
    _Currtem_spinBox = Signal(float)  # 更新当前温度
    _show_error_message = Signal(str)

class Action():
    def __init__(self, ui: QWidget, ms) -> None:
        self.ui = ui
        self.ms = ms

    def _draw_open(self):
        self.ui.draw_flag = True
        print("继续绘图", self.ui.draw_flag)

    def _draw_close(self):
        self.ui.draw_flag = False
        print("暂停绘图", self.ui.draw_flag)

    def _Clear_Button(self, value: bool): # 清洗按钮
        self.ui.state_open(self.ui.Clear_Button, not value)
        self.ui.Clear_Button.setEnabled(value)

    def _Collectbegin_Button(self, value: bool): # 采样按钮
        self.ui.state_open(self.ui.Collectbegin_Button, not value)
        self.ui.Collectbegin_Button.setEnabled(value)

    def _Clearroom_Button(self, value: bool): # 采样按钮
        self.ui.state_open(self.ui.Clearroom_Button, not value)
        self.ui.Clearroom_Button.setEnabled(value)

    def _statues_label(self, text: str):
        self.ui.statues_label.setText(text)

    def _ClearDraw(self):
        print("清除draw")
        self.ui.data = [[] for _ in range(self.ui.data_len)]
        self.ui.alldata = [[] for _ in range(self.ui.data_len)]
        # self.ui.redraw()
        # self.ui.update_table()

    def _print(self, time: int):
        self.ui.Worktime_spinBox.setValue(time)

    def _attendtime_spinBox(self, time: int):
        self.ui.attendtime_spinBox.setValue(time) #到达温度所需时间

    def _Currtem_spinBox(self, value: float):
        self.ui.Currtem_spinBox.setValue(value) #不断更新现在温度

    def _show_error_message(self, message):
        # 创建一个根窗口（不显示）
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        messagebox.showerror("错误", message)  # 弹出错误提示框


# 主窗口类
class GraphShowWindow(QWidget, Ui_Gragh_show):
    def __init__(self):
        super(GraphShowWindow, self).__init__()
        self.setupUi(self)  # 设置 UI 界面
        self.setWindowTitle("串口数据实时显示")
        self.ms = MySignals()
        self.a = Action(self, self.ms)
        self.initMS()

        self.lock = threading.Lock()
        self.data_len = len(g_var.sensors)
        self.now_data = 0
        self.now_num = 0
        self.data = [[] for _ in range(self.data_len)]
        self.alldata = [[] for _ in range(self.data_len)]

        # 初始化绘图
        self.plot_widget = pg.PlotWidget()
        self.Linegragh_Layout.addWidget(self.plot_widget)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')  # 设置绘图背景为白色
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time(单位:s)')
        # 启动事件过滤器来监听键盘事件
        self.plot_widget.installEventFilter(self)

        self.curves = []
        self.file_count = 1  # 用于追踪文件序号
        self.draw = None
        self.time_th = None
        self.draw_flag = False
        self.get_time = 60
        self._data_lines = dict()  # 已存在的绘图线
        self._data_colors = dict()  # 绘图颜色
        self._data_visible = g_var.sensors.copy() # 选择要看的传感器
        self.colors = self.generate_random_color_list(self.data_len)
        self.color_cycle = cycle(self.colors)

        # 初始化传感器数据表
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Sensor', 'Value'])
        self.model.itemChanged.connect(self.check_check_state)  # 连接项目更改信号
        self.Senser_stableView.setModel(self.model)
        self.Senser_stableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 初始化串口
        self.serial_setting()
        # 连接信号
        self.Auto_state = "idle"  # 初始状态为 "idle"
        self.Auto_Button.clicked.connect(self.Auto_Model) #电子鼻自动模式
        self.Clear_Button.clicked.connect(self.clear_data) # 基线清理阶段
        self.Collectbegin_Button.clicked.connect(self.start_serial) # 采集阶段
        self.Clearroom_Button.clicked.connect(self.clear_room)
        self.InitPos_Button.clicked.connect(self.Stra) #初始化位置
        self.Folder_Button.clicked.connect(self.savefolder) # 确认保存路径
        self.Stop_Button.clicked.connect(self.Stop)  # 全部暂停
        if g_var.Auto_falg == True:
            self.Autochoose_Button.clicked.connect(self.Autoinsample)  # 自动进样器

    def initMS(self):
        self.ms._draw_open.connect(self.a._draw_open)
        self.ms._draw_close.connect(self.a._draw_close)
        self.ms._Clear_Button.connect(self.a._Clear_Button)
        self.ms._Clearroom_Button.connect(self.a._Clearroom_Button)
        self.ms._Collectbegin_Button.connect(self.a._Collectbegin_Button)
        self.ms._statues_label.connect(self.a._statues_label)
        self.ms._ClearDraw.connect(self.a._ClearDraw)
        self.ms._print.connect(self.a._print)
        self.ms._attendtime_spinBox.connect(self.a._attendtime_spinBox)
        self.ms._Currtem_spinBox.connect(self.a._Currtem_spinBox)
        self.ms._show_error_message.connect(self.a._show_error_message)

    def serial_setting(self):
        if g_var.Auto_falg == True:
            if (g_var.Port_select2 == "" or g_var.Port_select == ""):
                self.statues_label.setText("串口初始化有问题")
            else:
                self.statues_label.setText("串口初始化成功")
            sconfig = [g_var.Port_select, g_var.Bund_select, g_var.Port_select2, g_var.Bund_select2]
            self.smng = mythread.SerialsMng(sconfig)
            self.ser = self.smng.ser_arr[0]
            self.ser.setSer(sconfig[0], sconfig[1])  # 设置串口及波特率
            self.ser1 = self.smng.ser_arr[1]
            self.ser1.setSer(sconfig[2], sconfig[3])  # 设置串口及波特率
            # 初始化操作对象
            self.Serialopea = SO.Serial1opea(self.ms, self.ser, self.ser1)
            self.open_serial1(self.Serialopea.GetSigal1)
        else:
            if (g_var.Port_select == ""):
                self.statues_label.setText("串口初始化有问题")
            else:
                self.statues_label.setText("串口初始化成功")
            sconfig = [g_var.Port_select, g_var.Bund_select]
            self.smng = mythread.SerialsMng(sconfig)
            self.ser = self.smng.ser_arr[0]
            self.ser.setSer(sconfig[0], sconfig[1])  # 设置串口及波特率
            self.Serialopea = SO.Serial1opea(self.ms, self.ser)
        # 完成串口信号初始化
        self.open_serial(self.process_data)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updata)
        self.timer.start(1000)  # 每秒更新一次
        # 开启线程
        self.time_th = SO.time_thread()  # 创建时间线程信号

    def open_serial(self, Signal): # 确保串口初始化
        if not self.ser.read_flag: # 如果串口存在
            d = self.ser.open(Signal, stock=1)
            print("控制串口初始化成功：", d)


    def open_serial1(self, Signal): # 确保串口初始化
        if not self.ser1.read_flag: # 如果串口存在
            d = self.ser1.open(Signal, stock=0,  flag=1) # flag=1: 16进制
            print("控制串口初始化成功：", d)

    def Stop(self):
        self.ser.write("00")
        if self.time_th:
            self.Serialopea.stop("Serialopea函数")
            self.time_th.stop("time_th函数")
        self.ms._draw_close.emit()
        self.button_init(True)
        if g_var.Auto_falg == True:
            self.ser.pause()
            self.ser1.pause()
        else:
            self.ser.pause()

    def SerStop(self):
        self.ser.stop()
        print("Gragn_show的ser 已关闭")
        if g_var.Auto_falg == True:
            self.ser1.stop()

    def Stra(self): # 暂停或者正式开始
        if g_var.Auto_falg == True:
            self.ser.resume()
            self.ser1.resume()
        else:
            self.ser.resume()
        # 1. 暂停串口采集和绘图
        if self.time_th:
            self.Serialopea.stop("time_th调用函数")
            self.Serialopea._running = False
        if self.draw:
            # self.draw.stop("draw")
            self.ms._draw_close.emit()
        # 初始化界面、线程、按钮
        self.plot_widget.repaint() # 刷新图形界面
        self.ms._Collectbegin_Button.emit(True)
        self.ms._Clear_Button.emit(True)
        self.ms._Clearroom_Button.emit(True)
        print("继续采集")
        g_var.sample_time = (int)(self.Sample_spinBox.value()) #采样时长
        g_var.exhaust_time = self.Cleartime_spinBox.value() # 洗气时常
        g_var.base_time = self.Basetime_spinBox.value() # 基线时长
        if g_var.Auto_falg == True:
            self.ser1.serialSend('0C', flag=True) #运动轴回到原点
        text = ("base_time:" + str(g_var.base_time) +
                ",sample_time:" + str(self.Sample_spinBox.value()) +
                ",exhaust_time:" + str(self.Cleartime_spinBox.value()) + ",flow_velocity:10\r\n")
        self.ser.write(text)
        self.ms._draw_close.emit()


    def Auto_Model(self):# 开始自动完成采集样本的三个阶段
        self.state_open(self.Auto_Button, True)
        self.Stra()
        self.Serialopea._running = True
        self.button_init(True)
        # 启动自动线程
        self.time_th.thread_loopfun(self.Serialopea.auto)

    def Autoinsample(self):
        self.Stra()
        self.Serialopea._running = True
        self.button_init(True)

        g_var.target_temp = self.Heattep_SpinBox.value() # 设置目标温度
        g_var.gettime = (int)(self.Sample_spinBox.value())
        g_var.cleartime = self.Cleartime_spinBox.value() # 洗气时常
        g_var.standtime = self.Standtime_spinBox.value() # 保持温度时常
        g_var.target_Sam = self.Simnum_spinBox.value() #样品个数
        time.sleep(1)
        # 到达1号位置
        self.ser1.serialSend("0A", g_var.posxyz[g_var.now_Sam + 1][0], g_var.posxyz[g_var.now_Sam + 1][1],
                             (int)(g_var.posxyz[g_var.now_Sam + 1][2] * 0.1), flag=True)  # 切换到下一个样品位置
        time.sleep(1)
        # 加热通道1
        self.ser1.serialSend(1, g_var.channal[1], int(self.Heattep_SpinBox.value()), flag=True)
        # 启动自动线程
        self.time_th.thread_loopfun(self.Serialopea.autosample)

    def clear_data(self): # 基线阶段
        self.button_init(True)
        self.Serialopea._running = True
        # 启动基线线程
        self.time_th.thread_loopfun(self.Serialopea.base_clear)

    def start_serial(self): # 开始采集
        self.button_init(True)
        self.Serialopea._running = True
        # 启动采样线程
        self.time_th.thread_loopfun(self.Serialopea.sample_collect)

    def clear_room(self): # 开始采集
        self.button_init(True)
        self.Serialopea._running = True
        # 启动清理线程
        self.time_th.thread_loopfun(self.Serialopea.room_clear)

    def button_init(self, value):
        print("按钮初始化")
        self.ms._Clear_Button.emit(value)
        self.ms._Collectbegin_Button.emit(value)
        self.ms._Clearroom_Button.emit(value)
        self.state_open(self.Auto_Button, False)
        self.Auto_state = "idle"


    def state_open(self, Button, state):
        if state == False:
            Button.setStyleSheet("")  # 设置成默认
        else:
            Button.setStyleSheet("background-color: #215b5d; color: white;")  # 激活状态颜色

    def process_data(self, data):
        self.now_data = 0
        # 解析数据
        if data != "" and (data[0] == "1"):
            if(data == "11"):
                self.ms._Clear_Button.emit(False)
            elif (data == "12"):
                self.ms._Clear_Button.emit(True)
        if data != "" and data[0] == "2":
            self.ser.pause()
            if (data == "21"):
                self.ms._Collectbegin_Button.emit(False)
                self.draw_flag = True
                self.data = [[] for _ in range(self.data_len)]
                self.alldata = [[] for _ in range(self.data_len)]
            elif (data == "22"):
                self.ms._Collectbegin_Button.emit(True)
                self.draw_flag = False
            self.ser.resume()
        if data != "" and (data[0] == "1" or data[0] == "2" or data[0] == "3" or data[0] == "4" or data[0] == "0"):
            if (data == "31"):
                self.ms._Clearroom_Button.emit(False)
            elif (data == "32"):
                self.ms._Clearroom_Button.emit(True)
            elif (data == "41"): # 自动模式
                self.state_open(self.Auto_Button, True)
            elif (data == "42"):
                self.state_open(self.Auto_Button, False)
            elif (data == "00"): # 暂停
                self.ms._Clear_Button.emit(True)
                self.ms._Collectbegin_Button.emit(True)
                self.ms._Clearroom_Button.emit(True)
                self.Stop()
        else:
            now_data = self.decode_data(data)
            if len(now_data) == self.data_len:
                now_data = [int(v) for v in now_data]
                for i, value in enumerate(now_data):
                    self.alldata[i].append(value)
                self.now_data = now_data

    def updata(self):
        if self.draw_flag == True and self.now_data != 0:
            now_data = self.now_data
            self.now_data = 0
            for i, value in enumerate(now_data):
                self.data[i].append(value)
                if len(self.data[i]) > 300:  # 限制数据长度
                    self.data[i].pop(0)
            self.redraw()  # 更新图表
            self.update_table()  # 更新表格


    def decode_data(self, data):
        # 在字符串 data 中查找所有与正则表达式 r'\d+' 匹配的子串，并以列表形式返回所有匹配结果。
        # return re.findall(r'\d+',data)
        # \d 表示任意一个数字字符（等价于 [0-9]）。
        # + 表示前面的字符（\d）出现一次或多次。
        pattern = r'(?P<name>[^=,]+)=(?P<value>\d+)'

        # 假设 data 是输入数据字符串
        self.pairs = {m.group('name'): int(m.group('value'))
                      for m in re.finditer(pattern, data)}

        # 只在 self.pairs 非空时进行后续操作
        if self.pairs:
            # 如果你仍需要按顺序的 16 个值：
            ordered_keys = list(self.pairs.keys())  # ['MQ3_1','MQ3_2',...,'base']

            # 确保顺序一致
            if g_var.sensors[0] != ordered_keys[0]:
                g_var.sensors = ordered_keys
                self._data_visible = g_var.sensors.copy()  # 选择要看的传感器

            values = [self.pairs[k] for k in ordered_keys]
            return values
        else:
            print("No valid pairs found")
            return []  # 或者返回一个空列表或其他处理逻辑

    def savefolder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder", "/")
        if folder_path:  # 如果用户选择了文件夹
            self.Folder_lineEdit.setText(folder_path)  # 设置 QLineEdit 的文本为选择的文件夹路径
        else:  # 如果用户取消了操作
            print("用户取消了选择")

    def savefile(self):
        try:
            with g_var.lock:
                # 筛选出选中的传感器数据
                selected_data = [self.alldata[g_var.sensors.index(sensor)] for sensor in self._data_visible]
            # 转置筛选后的数据
            transposed_data = list(map(list, zip(*selected_data)))
            # 假设 selected_data 是一个包含多个传感器数据的二维列表或数组
            selected_data_df = pd.DataFrame(transposed_data, columns=self._data_visible)  # 将数据转换为 DataFrame
            if self.Folder_lineEdit.text() == "":
                Tab_add.ADDTAB.save_text(selected_data_df)
            else:
                # 获取当前时间
                current_time = datetime.now()
                base_filename = current_time.strftime("%Y_%m_%d")  # 格式化为 YYYY_MM_DD
                file_path = os.path.join(self.Folder_lineEdit.text(), f"{base_filename}_{self.file_count}.txt")

                # 检查文件是否存在，并增加计数器
                while os.path.exists(file_path):
                    self.file_count += 1
                    file_path = os.path.join(self.Folder_lineEdit.text(), f"{base_filename}_{self.file_count}.txt")

                # 将 DataFrame 转换为文本字符串
                text_str = selected_data_df.to_csv(index=False, sep=' ')
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text_str)  # 保存为 TXT 文件

                print(f"Text file saved to {file_path}")

        except Exception as e:
            print("保存失败: " + str(e))

    def check_check_state(self, item):
        """
        检查并处理可复选项 item 的状态变化。
        """
        if item.column() == 0:
            sensor_name = item.text()
            checked = item.checkState() == Qt.CheckState.Checked

            if checked:
                if sensor_name not in self._data_visible:
                    self._data_visible.append(sensor_name)
            else:
                if sensor_name in self._data_visible:
                    self._data_visible.remove(sensor_name)
            self.redraw()

    def redraw(self):
        """
        Process data from store and prefer to draw.
        :return:
        """
        # 清空所有绘图线
        for line in self._data_lines.values():
            line.setData([], [])  # 清空数据
        # 更新绘图
        for sensor_name in self._data_visible:  # 只处理选中的传感器
            index = g_var.sensors.index(sensor_name)  # 获取传感器在 g_var.sensors 中的索引
            data_list = self.data[index]  # 获取对应的数据列表
            if data_list:  # 如果数据列表不为空
                if sensor_name in self._data_lines:
                    self._data_lines[sensor_name].setData(range(len(data_list)), data_list)  # 更新已存在的绘图线
                else:
                    self._data_lines[sensor_name] = self.plot_widget.plot(
                        range(len(data_list)),
                        data_list,
                        pen=pg.mkPen(self.get_currency_color(sensor_name), width=3),
                    )  # 创建新的绘图线
                    self.plot_widget.repaint() # 手动更新

    def update_table(self):
        # 先缓存所有的 QStandardItem 对象
        items = [(QStandardItem(), QStandardItem()) for _ in range(len(g_var.sensors))]  # 列出所有 item 对象
        for i, sensor_name in enumerate(g_var.sensors):
            value = self.data[i][-1] if self.data[i] else 0

            item_name, item_value = items[i]  # 获取已创建的 item 对象
            item_name.setText(sensor_name)  # 设置传感器名称作为文本
            item_name.setForeground(QBrush(QColor(self.get_currency_color(sensor_name))))  # 设置传感器名称颜色
            item_name.setCheckable(True)  # 设置为可复选框
            item_name.setEditable(False)  # 设置为不可编辑
            if sensor_name in self._data_visible:
                # 设置复选框状态为选中
                if item_name.checkState() != Qt.CheckState.Checked:
                    item_name.setCheckState(Qt.CheckState.Checked)

            item_value.setText(f"{value:.2f}")  # 设置传感器数据
            item_value.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)  # 设置文本对齐
            item_value.setEditable(False)  # 设置为不可编辑

        # 一次性更新所有数据
        for i, (item_name, item_value) in enumerate(items):
            self.model.setItem(i, 0, item_name)
            self.model.setItem(i, 1, item_value)

        # 设置列数为固定的 2
        if self.model.columnCount() != 2:
            self.model.setColumnCount(2)

    def generate_random_color_list(self, length):
        """生成一个指定长度的随机十六进制颜色代码列表"""
        return [self.generate_random_hex_color() for _ in range(length)]

    def get_currency_color(self, sensor):
        if sensor not in self._data_colors:
            self._data_colors[sensor] = next(self.color_cycle)

        return self._data_colors[sensor]

    def generate_random_hex_color(self):
        """生成一个随机的十六进制颜色代码"""
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def closeEvent(self, event):
        # 1. 停所有串口线程
        if hasattr(self, 'smng'): # 关闭所有串口
            for sop in self.smng.ser_arr:
                if hasattr(sop, 'read_flag'):
                    sop.read_flag = False
                    # 如果用了 QThread，也调 quit + wait
                    if hasattr(sop, 'thread') and sop.thread.isRunning():
                        sop.stop()
                        print("sop.stop()")
                        sop.thread.quit()
                        sop.thread.wait()
        if self.time_th:
            self.time_th.stop("time_th")
        # if self.draw:
        #     self.draw.stop("draw")
        if self.timer.isActive():
            self.timer.stop()  # 停止计时器

        event.accept()  # 允许窗口真正关闭

    def eventFilter(self, source, event):
        # 过滤来自plot_widget的事件
        if source is self.plot_widget and event.type() == QEvent.KeyPress:
            if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
                # 如果按下CTRL + S，调用保存函数
                self.savefile()
                return True  # 表示事件已处理
        return super().eventFilter(source, event)



if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = GraphShowWindow()
        window.show()
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n用户中断，程序结束。")
        sys.exit(0)