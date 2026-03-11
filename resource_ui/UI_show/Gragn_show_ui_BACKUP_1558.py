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
        self.ser = None
        self.get_time = 60
        self._data_lines = dict()  # 已存在的绘图线
        self._data_colors = dict()  # 绘图颜色
        self._data_visible = g_var.sensors.copy() # 选择要看的传感器，初始选择所有传感器
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
        self.Save_Button.clicked.connect(self.savefile) #保存文件
<<<<<<< HEAD
        g_var.sample_time = (int)(self.Sample_spinBox.value()) #采样时长
        g_var.exhaust_time = self.Cleartime_spinBox.value() # 洗气时常
        g_var.base_time = self.Basetime_spinBox.value() # 基线时长
=======
        self.ser = None
>>>>>>> 6153b490e701af03fc02cfcc3d1ef67ae7624264
        if g_var.Auto_falg == True:
            self.ser1 = None
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
        """初始化串口设置"""
        try:
            # 初始化串口管理器
            if g_var.Auto_falg:
                # 双串口模式
                if not all([g_var.Port_select, g_var.Bund_select, g_var.Port_select2, g_var.Bund_select2]):
                    self.statues_label.setText("串口初始化有问题")
                    logging.warning("双串口模式下缺少必要的串口配置")
                    return
                
                sconfig = [g_var.Port_select, g_var.Bund_select, g_var.Port_select2, g_var.Bund_select2]
                self.smng = mythread.SerialsMng(sconfig)
                
                # 初始化第一个串口
                self.ser = self.smng.ser_arr[0]
                self.ser.setSer(sconfig[0], sconfig[1])
                
                # 初始化第二个串口
                self.ser1 = self.smng.ser_arr[1]
                self.ser1.setSer(sconfig[2], sconfig[3])
                
                # 初始化操作对象
                self.Serialopea = SO.Serial1opea(self.ms, self.ser, self.ser1)
                
                # 打开第二个串口
                self.open_serial1(self.Serialopea.GetSigal1)
            else:
                # 单串口模式
                if not all([g_var.Port_select, g_var.Bund_select]):
                    self.statues_label.setText("串口初始化有问题")
                    logging.warning("单串口模式下缺少必要的串口配置")
                    return
                
                sconfig = [g_var.Port_select, g_var.Bund_select]
                self.smng = mythread.SerialsMng(sconfig)
                
                # 初始化串口
                self.ser = self.smng.ser_arr[0]
                self.ser.setSer(sconfig[0], sconfig[1])
                
                # 初始化操作对象
                self.Serialopea = SO.Serial1opea(self.ms, self.ser)
            
            # 打开主串口
            self.open_serial(self.process_data)
            
            # 初始化定时器
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.updata)
            self.timer.start(1000)  # 每秒更新一次
            
            # 初始化时间线程
            self.time_th = SO.time_thread()
            
            self.statues_label.setText("串口初始化成功")
            logging.info("串口初始化完成")
            
        except Exception as e:
            self.statues_label.setText("串口初始化失败")
            logging.error(f"串口初始化时出错: {e}")

    def open_serial(self, Signal): # 确保串口初始化
        if not self.ser.read_flag: # 如果串口存在
            d = self.ser.open(Signal, stock=1, slip=b'\n\r')
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

    def clear_room(self): # 开始清理
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
        """处理串口数据，根据数据类型执行不同的操作"""
        try:
            # 重置当前数据
            self.now_data = 0
            
            # 处理命令数据
            if data and data[0] in ["0", "1", "2", "3", "4"]:
                self._process_command_data(data)
            else:
                # 处理传感器数据
                self._process_sensor_data(data)
        except Exception as e:
            logging.error(f"处理数据时出错: {e}")
            
    def _process_command_data(self, data):
        """处理命令数据"""
        if data.startswith("1"):
            if data == "11":
                self.ms._Clear_Button.emit(False)
            elif data == "12":
                self.ms._Clear_Button.emit(True)
        
        elif data.startswith("2"):
            # 暂停串口以处理命令
            self.ser.pause()
            if data == "21":
                self.ms._Collectbegin_Button.emit(False)
                self.draw_flag = True
                # 重置数据列表
                self.data = [[] for _ in range(self.data_len)]
                self.alldata = [[] for _ in range(self.data_len)]
            elif data == "22":
                self.ms._Collectbegin_Button.emit(True)
                self.draw_flag = False
            # 恢复串口
            self.ser.resume()
        
        elif data.startswith("3"):
            if data == "31":
                self.ms._Clearroom_Button.emit(False)
            elif data == "32":
                self.ms._Clearroom_Button.emit(True)
        
        elif data.startswith("4"):
            if data == "41":  # 自动模式
                self.state_open(self.Auto_Button, True)
            elif data == "42":
                self.state_open(self.Auto_Button, False)
        
        elif data == "00":  # 暂停
            self.ms._Clear_Button.emit(True)
            self.ms._Collectbegin_Button.emit(True)
            self.ms._Clearroom_Button.emit(True)
            self.Stop()
    
    def _process_sensor_data(self, data):
        """处理传感器数据"""
        now_data = self.decode_data(data)
        if len(now_data) == self.data_len:
            # 转换数据类型
            now_data = [int(v) for v in now_data]
            # 保存到全部数据
            with self.lock:
                for i, value in enumerate(now_data):
                    self.alldata[i].append(value)
            # 更新当前数据
            self.now_data = now_data

    def updata(self):
        """更新UI数据，包括图表和表格"""
        if self.draw_flag and self.now_data:
            # 复制当前数据并重置
            now_data = self.now_data
            self.now_data = 0
            
            # 更新数据列表
            for i, value in enumerate(now_data):
                self.data[i].append(value)
                # 限制数据长度，避免内存占用过大
                if len(self.data[i]) > 300:
                    self.data[i].pop(0)
            
            # 异步更新UI，避免阻塞主线程
            QTimer.singleShot(0, self._update_ui)
    
    def _update_ui(self):
        """异步更新UI，提高性能"""
        try:
            # 更新图表
            self.redraw()
            # 更新表格
            self.update_table()
        except Exception as e:
            logging.error(f"更新UI时出错: {e}")


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
            # 去除传感器名称中的空格
            ordered_keys = [key.strip() for key in self.pairs.keys() if key.strip()]
            # 重新构建pairs字典，确保键不包含空格
            self.pairs = {key: self.pairs[key] for key in ordered_keys}

            # 确保顺序一致
            if not g_var.sensors or (g_var.sensors and g_var.sensors[0] != ordered_keys[0]):
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
        """保存数据到文件"""
        try:
            # 检查保存路径
            save_dir = self.Folder_lineEdit.text()
            if not save_dir:
                self.ms._show_error_message.emit("请先设置保存路径")
                logging.warning("保存路径未设置")
                return
            
            # 确保保存目录存在
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 复制数据以避免并发问题
            with self.lock:
                # 筛选出选中的传感器数据
                selected_data = []
                selected_sensors = []
                # 确保g_var.sensors不为空
                if not g_var.sensors:
                    logging.error("传感器列表为空")
                    self.ms._show_error_message.emit("传感器列表为空，无法保存")
                    return
                
                # 遍历所有传感器，确保数据和传感器名称一一对应
                for i, sensor in enumerate(g_var.sensors):
                    # 检查传感器是否在选中列表中
                    if sensor in self._data_visible:
                        try:
                            # 去除传感器名称中的空格
                            sensor = sensor.strip()
                            if sensor:
                                selected_data.append(self.alldata[i].copy())
                                selected_sensors.append(sensor)
                        except Exception as e:
                            logging.warning(f"处理传感器 '{sensor}' 时出错: {e}")
            
            # 检查数据是否为空
            if not selected_data or any(len(data) == 0 for data in selected_data):
                logging.warning("没有数据可保存")
                return
            
            # 检查所有数据长度是否一致
            data_lengths = [len(data) for data in selected_data]
            if len(set(data_lengths)) > 1:
                logging.error(f"数据长度不一致: {data_lengths}")
                self.ms._show_error_message.emit("数据长度不一致，无法保存")
                return
            
            # 转置数据
            transposed_data = list(map(list, zip(*selected_data)))
            
            # 检查转置后的数据长度
            logging.info(f"转置前数据长度: {len(selected_data)}, 转置后数据长度: {len(transposed_data)}")
            logging.info(f"传感器数量: {len(selected_sensors)}, 列名数量: {len(selected_sensors)}")
            
            # 创建DataFrame
            try:
                selected_data_df = pd.DataFrame(transposed_data, columns=selected_sensors)
                logging.info(f"成功创建DataFrame，形状: {selected_data_df.shape}")
            except Exception as e:
                logging.error(f"创建DataFrame时出错: {e}")
                logging.error(f"转置后数据长度: {len(transposed_data)}")
                logging.error(f"列名数量: {len(selected_sensors)}")
                if transposed_data:
                    logging.error(f"第一行数据长度: {len(transposed_data[0])}")
                self.ms._show_error_message.emit(f"创建数据表格时出错: {e}")
                return
            
            # 生成文件名
            current_time = datetime.now()
            base_filename = current_time.strftime("%Y_%m_%d")
            
            # 查找可用的文件名
            file_path = os.path.join(save_dir, f"{base_filename}_{self.file_count}")
            while os.path.exists(file_path):
                self.file_count += 1
                file_path = os.path.join(save_dir, f"{base_filename}_{self.file_count}")
            
            # 保存文件
            try:
                Tab_add.ADDTAB.save_text(selected_data_df, file_path)
                logging.info(f"文件保存成功: {file_path}")
            except Exception as e:
                logging.error(f"保存文件时出错: {e}")
                self.ms._show_error_message.emit(f"保存文件时出错: {e}")
                return
            logging.info(f"文件保存成功: {file_path}")
            
        except Exception as e:
            logging.error(f"保存文件时出错: {e}")

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
        """更新图表显示"""
        # 只处理选中的传感器
        for sensor_name in self._data_visible:
            try:
                # 获取传感器索引
                index = g_var.sensors.index(sensor_name)
                data_list = self.data[index]
                
                if data_list:
                    # 检查是否已存在绘图线
                    if sensor_name in self._data_lines:
                        # 更新已存在的绘图线
                        self._data_lines[sensor_name].setData(range(len(data_list)), data_list)
                    else:
                        # 创建新的绘图线
                        pen_color = self.get_currency_color(sensor_name)
                        self._data_lines[sensor_name] = self.plot_widget.plot(
                            range(len(data_list)),
                            data_list,
                            pen=pg.mkPen(pen_color, width=3),
                        )
            except Exception as e:
                logging.error(f"更新图表时出错: {e}")
        
        # 移除不再可见的传感器绘图线
        sensors_to_remove = []
        for sensor_name in self._data_lines:
            if sensor_name not in self._data_visible:
                sensors_to_remove.append(sensor_name)
        
        for sensor_name in sensors_to_remove:
            try:
                # 移除绘图线
                line = self._data_lines.pop(sensor_name, None)
                if line:
                    self.plot_widget.removeItem(line)
            except Exception as e:
                logging.error(f"移除绘图线时出错: {e}")

    def update_table(self):
        """更新传感器数据表格"""
        try:
            # 检查模型是否需要重置
            if self.model.rowCount() != len(g_var.sensors):
                self.model.setRowCount(len(g_var.sensors))
            
            # 更新表格数据
            for i, sensor_name in enumerate(g_var.sensors):
                # 获取当前值
                value = self.data[i][-1] if self.data[i] else 0
                
                # 获取或创建名称项
                name_item = self.model.item(i, 0)
                if not name_item:
                    name_item = QStandardItem()
                    name_item.setCheckable(True)
                    name_item.setCheckState(Qt.CheckState.Checked)  # 默认选中所有传感器
                    name_item.setEditable(False)
                    self.model.setItem(i, 0, name_item)
                
                # 更新名称项
                if name_item.text() != sensor_name:
                    name_item.setText(sensor_name)
                    name_item.setForeground(QBrush(QColor(self.get_currency_color(sensor_name))))
                
                # 更新复选状态
                should_be_checked = sensor_name in self._data_visible
                if name_item.checkState() != (Qt.CheckState.Checked if should_be_checked else Qt.CheckState.Unchecked):
                    name_item.setCheckState(Qt.CheckState.Checked if should_be_checked else Qt.CheckState.Unchecked)
                
                # 获取或创建值项
                value_item = self.model.item(i, 1)
                if not value_item:
                    value_item = QStandardItem()
                    value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    value_item.setEditable(False)
                    self.model.setItem(i, 1, value_item)
                
                # 更新值项
                new_value_text = f"{value:.2f}"
                if value_item.text() != new_value_text:
                    value_item.setText(new_value_text)
            
            # 确保列数正确
            if self.model.columnCount() != 2:
                self.model.setColumnCount(2)
        except Exception as e:
            logging.error(f"更新表格时出错: {e}")

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
        try:
            # 1. 停止定时器（必须在主线程中操作）
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()  # 停止计时器
                print("定时器已停止")
            
            # 2. 停止时间线程
            if hasattr(self, 'time_th') and self.time_th:
                # 只设置标志，让线程自然结束，避免跨线程操作
                self.time_th._running = False
                self.time_th._stop_evt.set()
                print("时间线程已停止")
            
            # 3. 停止串口操作
            if hasattr(self, 'Serialopea') and self.Serialopea:
                self.Serialopea._running = False
                print("串口操作已停止")
            
            # 4. 关闭所有串口
            if hasattr(self, 'smng'): # 关闭所有串口
                for sop in self.smng.ser_arr:
                    if hasattr(sop, 'read_flag'):
                        sop.read_flag = False
                        # 如果用了 QThread，也调 quit + wait
                        if hasattr(sop, 'thread') and hasattr(sop.thread, 'isRunning') and sop.thread.isRunning():
                            try:
                                sop.stop()
                                print("sop.stop()")
                                if hasattr(sop.thread, 'quit'):
                                    sop.thread.quit()
                                if hasattr(sop.thread, 'wait'):
                                    sop.thread.wait(1000)  # 等待最多1秒
                            except Exception as e:
                                print(f"关闭串口线程时出错: {e}")
            
            # 5. 给线程一点时间来清理
            import time
            time.sleep(0.1)
            
            print("所有资源已清理")
            event.accept()  # 允许窗口真正关闭
        except Exception as e:
            print(f"关闭事件处理时出错: {e}")
            event.accept()  # 即使出错也要允许窗口关闭

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