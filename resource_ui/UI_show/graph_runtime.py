"""重构版图形页的运行时辅助模块。

这个文件抽离了图形页运行时的公共能力，例如绘图数据状态、
串口控制器初始化和页面运行环境准备，减少界面类本身的耦合。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from itertools import cycle
from typing import Dict, List, Optional

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QHeaderView

import global_var as legacy_global
import tool.Serial_opea as SO
import tool.serial_thread as mythread


@dataclass
class GraphDataState:
    data_len: int
    now_data: int | List[int] = 0
    now_num: int = 0
    data: List[List[int]] = field(init=False)
    alldata: List[List[int]] = field(init=False)
    curves: List = field(default_factory=list)
    file_count: int = 1
    draw: Optional[object] = None
    time_thread: Optional[SO.time_thread] = None
    draw_flag: bool = False
    get_time: int = 60
    data_lines: Dict[str, object] = field(default_factory=dict)
    data_colors: Dict[str, str] = field(default_factory=dict)
    visible_sensors: List[str] = field(default_factory=list)

    # 初始化数据缓存结构。
    def __post_init__(self) -> None:
        self.data = [[] for _ in range(self.data_len)]
        self.alldata = [[] for _ in range(self.data_len)]


class GraphSerialController:
    # 初始化图形页串口控制器。
    def __init__(self, window, signals) -> None:
        self.window = window
        self.signals = signals
        self.serial_manager = None
        self.primary_serial = None
        self.secondary_serial = None
        self.serial_operator = None
        self.timer = None

    # 按当前配置初始化串口与定时器。
    def initialize(self) -> bool:
        try:
            if legacy_global.app_state.auto_mode:
                if not all(
                    [
                        legacy_global.app_state.serial.primary_port,
                        legacy_global.app_state.serial.primary_baud,
                        legacy_global.app_state.serial.secondary_port,
                        legacy_global.app_state.serial.secondary_baud,
                    ]
                ):
                    self.window.statues_label.setText("串口初始化有问题")
                    logging.warning("双串口模式下缺少必要串口配置")
                    return False

                serial_config = [
                    legacy_global.app_state.serial.primary_port,
                    legacy_global.app_state.serial.primary_baud,
                    legacy_global.app_state.serial.secondary_port,
                    legacy_global.app_state.serial.secondary_baud,
                ]
                self.serial_manager = mythread.SerialsMng(serial_config)
                self.primary_serial = self.serial_manager.ser_arr[0]
                self.primary_serial.setSer(serial_config[0], serial_config[1])
                self.secondary_serial = self.serial_manager.ser_arr[1]
                self.secondary_serial.setSer(serial_config[2], serial_config[3])
                self.serial_operator = SO.Serial1opea(self.signals, self.primary_serial, self.secondary_serial)
                self._open_secondary_serial(self.serial_operator.GetSigal1)
            else:
                if not all([legacy_global.app_state.serial.primary_port, legacy_global.app_state.serial.primary_baud]):
                    self.window.statues_label.setText("串口初始化有问题")
                    logging.warning("单串口模式下缺少必要串口配置")
                    return False

                serial_config = [legacy_global.app_state.serial.primary_port, legacy_global.app_state.serial.primary_baud]
                self.serial_manager = mythread.SerialsMng(serial_config)
                self.primary_serial = self.serial_manager.ser_arr[0]
                self.primary_serial.setSer(serial_config[0], serial_config[1])
                self.serial_operator = SO.Serial1opea(self.signals, self.primary_serial)

            self._open_primary_serial(self.window.process_data)
            self.timer = QTimer(self.window)
            self.timer.timeout.connect(self.window.updata)
            self.timer.start(1000)
            self.window.statues_label.setText("串口初始化成功")
            logging.info("串口初始化完成")
            return True
        except Exception as exc:
            self.window.statues_label.setText("串口初始化失败")
            logging.error("串口初始化时出错: %s", exc)
            return False

    # 打开主串口并注册数据回调。
    def _open_primary_serial(self, callback) -> None:
        if self.primary_serial and not self.primary_serial.read_flag:
            result = self.primary_serial.open(callback, stock=1, slip=b"\n\r")
            print("控制串口初始化成功：", result)

    # 打开副串口并注册控制回调。
    def _open_secondary_serial(self, callback) -> None:
        if self.secondary_serial and not self.secondary_serial.read_flag:
            result = self.secondary_serial.open(callback, stock=0, flag=1)
            print("控制串口初始化成功：", result)

    # 关闭串口和定时器资源。
    def shutdown(self) -> None:
        if self.timer and self.timer.isActive():
            self.timer.stop()

        if self.serial_operator:
            self.serial_operator._running = False

        if self.serial_manager:
            for serial_port in self.serial_manager.ser_arr:
                if hasattr(serial_port, "read_flag"):
                    serial_port.read_flag = False
                if hasattr(serial_port, "stop"):
                    try:
                        serial_port.stop()
                    except Exception as exc:
                        logging.error("关闭串口时出错: %s", exc)


# 初始化图形页运行时需要的绘图和表格状态。
def initialize_graph_runtime(window) -> GraphDataState:
    state = GraphDataState(data_len=len(legacy_global.app_state.sensor_names))
    state.visible_sensors = list(legacy_global.app_state.sensor_names)
    window.lock = threading.Lock()

    plot_widget = pg.PlotWidget()
    window.Linegragh_Layout.addWidget(plot_widget)
    plot_widget.showGrid(x=True, y=True)
    plot_widget.setBackground("w")
    plot_widget.setLabel("left", "Value")
    plot_widget.setLabel("bottom", "Time(单位:s)")
    plot_widget.installEventFilter(window)
    window.plot_widget = plot_widget

    window.model = QStandardItemModel()
    window.model.setHorizontalHeaderLabels(["Sensor", "Value"])
    window.model.itemChanged.connect(window.check_check_state)
    window.Senser_stableView.setModel(window.model)
    window.Senser_stableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    colors = window.generate_random_color_list(state.data_len)
    window.color_cycle = cycle(colors)
    return state
