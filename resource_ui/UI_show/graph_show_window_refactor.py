"""新的图形显示页面实现。

这个文件提供重构后的 ``GraphShowWindow``，由 ``running.py`` 直接调用。
目标是在尽量保持原有界面行为的前提下，把状态管理和串口初始化逻辑
迁移到新的结构化模块中，降低与旧代码的耦合。
"""


import logging
import os
import random
import re
import time
from datetime import datetime

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QStandardItem
from PySide6.QtWidgets import QApplication, QFileDialog, QWidget

import global_var as g_var
import resource_ui.alg_puifile.pic_tab_add as Tab_add
import tool.Serial_opea as SO
from resource_ui.UI_show.graph_runtime import GraphSerialController, initialize_graph_runtime

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

if g_var.app_state.auto_mode:
    from resource_ui.ui_pfile.ChooseAndShow import Ui_Gragh_show
else:
    from resource_ui.ui_pfile.ChooseAndShow_1 import Ui_Gragh_show


class MySignals(QObject):
    draw_open = Signal()
    draw_close = Signal()
    clear_button = Signal(bool)
    collect_button = Signal(bool)
    clearroom_button = Signal(bool)
    status_label = Signal(str)
    clear_draw = Signal()
    worktime_value = Signal(int)
    attendtime_value = Signal(int)
    current_temp_value = Signal(float)
    error_message = Signal(str)
    _draw_open = Signal()
    _draw_close = Signal()
    _Clear_Button = Signal(bool)
    _Collectbegin_Button = Signal(bool)
    _Clearroom_Button = Signal(bool)
    _statues_label = Signal(str)
    _ClearDraw = Signal()
    _print = Signal(int)
    _Currtem_spinBox = Signal(float)
    _attendtime_spinBox = Signal(int)
    _show_error_message = Signal(str)
    task_finished = Signal()


class Action:
    # 初始化界面动作分发器。
    def __init__(self, ui: QWidget) -> None:
        self.ui = ui

    # 根据当前任务状态更新流程按钮。
    def _update_task_button(self, button, value: bool):
        if self.ui._task_running and not value:
            if self.ui._active_running_button is not None and self.ui._active_running_button is not button:
                self.ui.state_open(self.ui._active_running_button, False)
            self.ui._active_running_button = button
            self.ui._set_running_state(True)
            return
        is_active_button = self.ui._task_running and self.ui._active_running_button is button
        if is_active_button and value:
            self.ui.state_open(button, False)
            self.ui._active_running_button = None

    # 打开绘图状态。
    def draw_open(self):
        self.ui.draw_flag = True

    # 关闭绘图状态。
    def draw_close(self):
        self.ui.draw_flag = False

    # 设置清空按钮状态。
    def clear_button(self, value: bool):
        self._update_task_button(self.ui.Clear_Button, value)

    # 设置采集按钮状态。
    def collect_button(self, value: bool):
        self._update_task_button(self.ui.Collectbegin_Button, value)

    # 设置清洗按钮状态。
    def clearroom_button(self, value: bool):
        self._update_task_button(self.ui.Clearroom_Button, value)

    # 更新状态标签文本。
    def status_label(self, text: str):
        self.ui.statues_label.setText(text)

    # 清空当前绘图缓存。
    def clear_draw(self):
        self.ui.data = [[] for _ in range(self.ui.data_len)]
        self.ui.alldata = [[] for _ in range(self.ui.data_len)]

    # 更新工作时间控件数值。
    def worktime_value(self, value: int):
        self.ui.Worktime_spinBox.setValue(value)

    # 更新等待时间控件数值。
    def attendtime_value(self, value: int):
        self.ui.attendtime_spinBox.setValue(value)

    # 更新当前温度控件数值。
    def current_temp_value(self, value: float):
        self.ui.Currtem_spinBox.setValue(value)

    # 弹出错误消息框。
    def error_message(self, message: str):
        from tkinter import Tk, messagebox

        root = Tk()
        root.withdraw()
        messagebox.showerror("错误", message)
        root.destroy()


class GraphShowWindow(QWidget, Ui_Gragh_show):
    # 初始化重构版图形显示页面。
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("串口数据实时显示")

        self.ms = MySignals()
        self.action = Action(self)
        self._connect_internal_signals()

        runtime = initialize_graph_runtime(self)
        self.data_len = runtime.data_len
        self.now_data = runtime.now_data
        self.now_num = runtime.now_num
        self.data = runtime.data
        self.alldata = runtime.alldata
        self.curves = runtime.curves
        self.file_count = runtime.file_count
        self.draw = runtime.draw
        self.time_th = runtime.time_thread
        self.draw_flag = runtime.draw_flag
        self.get_time = runtime.get_time
        self._data_lines = runtime.data_lines
        self._data_colors = runtime.data_colors
        self._data_visible = runtime.visible_sensors

        self.ser = None
        self.ser1 = None
        self.Serialopea = None
        self.timer = None
        self.smng = None
        self.Auto_state = "idle"

        self.serial_controller = GraphSerialController(self, self.ms)
        self.serial_setting()
        self._task_running = False
        self._active_running_button = None
        self._managed_widgets = [
            widget
            for widget in [
                getattr(self, "Auto_Button", None),
                getattr(self, "Clear_Button", None),
                getattr(self, "Collectbegin_Button", None),
                getattr(self, "Clearroom_Button", None),
                getattr(self, "InitPos_Button", None),
                getattr(self, "Sample_spinBox", None),
                getattr(self, "Cleartime_spinBox", None),
                getattr(self, "Basetime_spinBox", None),
                getattr(self, "Heattep_SpinBox", None),
                getattr(self, "Simnum_spinBox", None),
            ]
            if widget is not None
        ]
        if g_var.app_state.auto_mode:
            self._managed_widgets.append(self.Autochoose_Button)

        self.Auto_Button.clicked.connect(self.Auto_Model)
        self.Clear_Button.clicked.connect(self.clear_data)
        self.Collectbegin_Button.clicked.connect(self.start_serial)
        self.Clearroom_Button.clicked.connect(self.clear_room)
        self.InitPos_Button.clicked.connect(self.Stra)
        self.Folder_Button.clicked.connect(self.savefolder)
        self.Stop_Button.clicked.connect(self.Stop)
        self.Save_Button.clicked.connect(self.savefile)
        if g_var.app_state.auto_mode:
            self.Autochoose_Button.clicked.connect(self.Autoinsample)

        self._sync_sample_config()
        self._set_running_state(False)

    # 连接页面内部信号与动作处理函数。
    def _connect_internal_signals(self):
        self.ms.draw_open.connect(self.action.draw_open)
        self.ms.draw_close.connect(self.action.draw_close)
        self.ms.clear_button.connect(self.action.clear_button)
        self.ms.collect_button.connect(self.action.collect_button)
        self.ms.clearroom_button.connect(self.action.clearroom_button)
        self.ms.status_label.connect(self.action.status_label)
        self.ms.clear_draw.connect(self.action.clear_draw)
        self.ms.worktime_value.connect(self.action.worktime_value)
        self.ms.attendtime_value.connect(self.action.attendtime_value)
        self.ms.current_temp_value.connect(self.action.current_temp_value)
        self.ms.error_message.connect(self.action.error_message)
        self.ms._draw_open.connect(self.action.draw_open)
        self.ms._draw_close.connect(self.action.draw_close)
        self.ms._Clear_Button.connect(self.action.clear_button)
        self.ms._Collectbegin_Button.connect(self.action.collect_button)
        self.ms._Clearroom_Button.connect(self.action.clearroom_button)
        self.ms._statues_label.connect(self.action.status_label)
        self.ms._ClearDraw.connect(self.action.clear_draw)
        self.ms._Currtem_spinBox.connect(self.action.current_temp_value)
        self.ms._attendtime_spinBox.connect(self.action.attendtime_value)
        self.ms._show_error_message.connect(self.action.error_message)
        self.ms.task_finished.connect(self._finish_running_state)

    # 将界面上的采样配置同步到全局状态。
    def _sync_sample_config(self):
        g_var.app_state.set_sample_config(
            sample_time=int(self.Sample_spinBox.value()),
            exhaust_time=self.Cleartime_spinBox.value(),
            base_time=self.Basetime_spinBox.value(),
        )

    # 初始化图形页串口相关对象。
    def serial_setting(self):
        if g_var.app_state.demo_mode:
            self.Serialopea = SO.Serial1opea(self.ms, None, None)
            self.time_th = SO.time_thread()
            self.action.status_label(f"演示模式已开启，每步等待 {g_var.app_state.demo_step_seconds} 秒")
            return
        if self.serial_controller.initialize():
            self.smng = self.serial_controller.serial_manager
            self.ser = self.serial_controller.primary_serial
            self.ser1 = self.serial_controller.secondary_serial
            self.Serialopea = self.serial_controller.serial_operator
            self.timer = self.serial_controller.timer
            self.time_th = SO.time_thread()

    # 统一设置页面是否处于运行中状态。
    def _set_running_state(self, running: bool):
        for widget in self._managed_widgets:
            if hasattr(widget, "setStyleSheet"):
                widget.setStyleSheet("")
            widget.setEnabled(not running)
        if running and self._active_running_button is not None:
            self._active_running_button.setEnabled(True)
            self.state_open(self._active_running_button, True)
        self.Stop_Button.setEnabled(True)

    # 启动一个实验流程并锁定其它按钮。
    def _start_task(self, task, active_button=None):
        if self._task_running or not self.Serialopea or not self.time_th:
            return
        def task_with_cleanup():
            try:
                task()
            finally:
                self.ms.task_finished.emit()

        self.Serialopea._running = True
        self._task_running = True
        self._active_running_button = active_button
        self._set_running_state(True)
        self.time_th.thread_loopfun(task_with_cleanup)

    # 恢复运行态按钮和状态标记。
    def _finish_running_state(self):
        self._task_running = False
        if self._active_running_button is not None:
            self.state_open(self._active_running_button, False)
            self._active_running_button = None
        self._set_running_state(False)

    # 停止当前采集流程。
    def Stop(self):
        if not self.ser and not g_var.app_state.demo_mode:
            return
        if self.ser:
            self.ser.write("00")
        if self.ser1:
            try:
                self.ser1.serialSend("00", flag=True)
            except Exception:
                pass
        if self.time_th:
            self.Serialopea.stop("Serialopea")
            self.time_th.stop("time_th")
        self.ms.draw_close.emit()
        self.button_init(True)
        if self.ser:
            self.ser.pause()
        if g_var.app_state.auto_mode and self.ser1:
            self.ser1.pause()

    # 彻底关闭串口连接。
    def SerStop(self):
        if self.ser:
            self.ser.stop()
        if g_var.app_state.auto_mode and self.ser1:
            self.ser1.stop()

    # 下发开始前的基础串口指令。
    def Stra(self):
        if not self.ser and not g_var.app_state.demo_mode:
            return
        if self.ser:
            self.ser.resume()
        if g_var.app_state.auto_mode and self.ser1:
            self.ser1.resume()

        if self.time_th:
            self.Serialopea.stop("time_th callback")
            self.Serialopea._running = False
        if self.draw:
            self.ms.draw_close.emit()

        self.plot_widget.repaint()
        self.ms.collect_button.emit(True)
        self.ms.clear_button.emit(True)
        self.ms.clearroom_button.emit(True)
        self._sync_sample_config()
        if g_var.app_state.auto_mode and self.ser1:
            self.ser1.serialSend("0C", flag=True)

        text = (
            f"base_time:{g_var.app_state.experiment.base_time},"
            f"sample_time:{self.Sample_spinBox.value()},"
            f"exhaust_time:{self.Cleartime_spinBox.value()},flow_velocity:10\r\n"
        )
        if self.ser:
            self.ser.write(text)
        self.ms.draw_close.emit()

    # 启动自动模式流程。
    def Auto_Model(self):
        self.Stra()
        self._start_task(self.Serialopea.auto, self.Auto_Button)

    # 启动自动进样流程。
    def Autoinsample(self):
        self.Stra()
        self.Serialopea._running = True

        heat_spinbox = getattr(self, "Heattep_SpinBox", None)
        sample_count_spinbox = getattr(self, "Simnum_spinBox", None)

        if heat_spinbox is not None:
            g_var.app_state.set_target_temp(heat_spinbox.value())
        if sample_count_spinbox is not None:
            g_var.app_state.set_target_sample_count(sample_count_spinbox.value())

        if self.ser1:
            time.sleep(1)
            self.ser1.serialSend(
                "0A",
                g_var.app_state.positions[g_var.app_state.current_sample_index + 1][0],
                g_var.app_state.positions[g_var.app_state.current_sample_index + 1][1],
                int(g_var.app_state.positions[g_var.app_state.current_sample_index + 1][2] * 0.1),
                flag=True,
            )
            time.sleep(1)
            self.ser1.serialSend(1, g_var.app_state.channels[1], int(g_var.app_state.target_temp), flag=True)
        self._start_task(self.Serialopea.autosample, getattr(self, "Autochoose_Button", self.Auto_Button))

    # 执行基线清空流程。
    def clear_data(self):
        self._start_task(self.Serialopea.base_clear, self.Clear_Button)

    # 启动采样采集流程。
    def start_serial(self):
        self._start_task(self.Serialopea.sample_collect, self.Collectbegin_Button)

    # 执行清洗气路流程。
    def clear_room(self):
        self._start_task(self.Serialopea.room_clear, self.Clearroom_Button)

    # 统一恢复按钮初始状态。
    def button_init(self, value: bool):
        self.ms.clear_button.emit(value)
        self.ms.collect_button.emit(value)
        self.ms.clearroom_button.emit(value)
        self.Auto_state = "idle"
        self._finish_running_state()

    # 切换当前运行按钮的高亮状态。
    def state_open(self, button, state: bool):
        if not state:
            button.setStyleSheet("")
        else:
            button.setStyleSheet(
                "background-color: #17c9cf; "
                "color: #ffffff; "
                "border: 2px solid #0f9fa5; "
                "font-weight: 700;"
            )

    # 分发收到的串口数据。
    def process_data(self, data):
        try:
            self.now_data = 0
            if data and data[0] in ["0", "1", "2", "3", "4"]:
                self._process_command_data(data)
            else:
                self._process_sensor_data(data)
        except Exception as exc:
            logging.error("处理数据时出错: %s", exc)

    # 处理控制类串口指令。
    def _process_command_data(self, data):
        if data.startswith("1"):
            if data == "11":
                self.ms.clear_button.emit(False)
            elif data == "12":
                self.ms.clear_button.emit(True)
        elif data.startswith("2"):
            self.ser.pause()
            if data == "21":
                self.ms.collect_button.emit(False)
                self.draw_flag = True
                self.data = [[] for _ in range(self.data_len)]
                self.alldata = [[] for _ in range(self.data_len)]
            elif data == "22":
                self.ms.collect_button.emit(True)
                self.draw_flag = False
            self.ser.resume()
        elif data.startswith("3"):
            if data == "31":
                self.ms.clearroom_button.emit(False)
            elif data == "32":
                self.ms.clearroom_button.emit(True)
        elif data.startswith("4"):
            if data in ["41", "42"]:
                self.state_open(self.Auto_Button, False)
        elif data == "00":
            self.ms.clear_button.emit(True)
            self.ms.collect_button.emit(True)
            self.ms.clearroom_button.emit(True)
            self.Stop()

    # 处理传感器数值数据。
    def _process_sensor_data(self, data):
        now_data = self.decode_data(data)
        if len(now_data) == self.data_len:
            now_data = [int(v) for v in now_data]
            with self.lock:
                for i, value in enumerate(now_data):
                    self.alldata[i].append(value)
            self.now_data = now_data

    # 定时刷新绘图数据缓存。
    def updata(self):
        if self.draw_flag and self.now_data:
            now_data = self.now_data
            self.now_data = 0
            for i, value in enumerate(now_data):
                self.data[i].append(value)
                if len(self.data[i]) > 300:
                    self.data[i].pop(0)
            QTimer.singleShot(0, self._update_ui)

    # 在主线程中刷新图表和表格。
    def _update_ui(self):
        try:
            self.redraw()
            self.update_table()
        except Exception as exc:
            logging.error("更新 UI 时出错: %s", exc)

    # 解析原始串口文本为传感器数值。
    def decode_data(self, data):
        pattern = r"(?P<name>[^=,]+)=(?P<value>\d+)"
        pairs = {m.group("name"): int(m.group("value")) for m in re.finditer(pattern, data)}
        if not pairs:
            return []

        ordered_keys = [key.strip() for key in pairs.keys() if key.strip()]
        pairs = {key: pairs[key] for key in ordered_keys}
        if not g_var.app_state.sensor_names or (
            g_var.app_state.sensor_names and g_var.app_state.sensor_names[0] != ordered_keys[0]
        ):
            g_var.app_state.update_sensor_names(ordered_keys)
            self._data_visible = g_var.app_state.sensor_names.copy()

        return [pairs[k] for k in ordered_keys]

    # 选择数据保存目录。
    def savefolder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder", "/")
        if folder_path:
            self.Folder_lineEdit.setText(folder_path)

    # 将当前数据保存到文件。
    def savefile(self):
        try:
            save_dir = self.Folder_lineEdit.text()
            if not save_dir:
                self.ms.error_message.emit("请先设置保存路径")
                return

            os.makedirs(save_dir, exist_ok=True)
            with self.lock:
                selected_data = []
                selected_sensors = []
                if not g_var.app_state.sensor_names:
                    self.ms.error_message.emit("传感器列表为空，无法保存")
                    return

                for i, sensor in enumerate(g_var.app_state.sensor_names):
                    if sensor in self._data_visible:
                        sensor = sensor.strip()
                        if sensor:
                            selected_data.append(self.alldata[i].copy())
                            selected_sensors.append(sensor)

            if not selected_data or any(len(item) == 0 for item in selected_data):
                return

            data_lengths = [len(item) for item in selected_data]
            if len(set(data_lengths)) > 1:
                self.ms.error_message.emit("数据长度不一致，无法保存")
                return

            selected_data_df = pd.DataFrame(list(map(list, zip(*selected_data))), columns=selected_sensors)
            base_filename = datetime.now().strftime("%Y_%m_%d")
            file_path = os.path.join(save_dir, f"{base_filename}_{self.file_count}")
            while os.path.exists(file_path):
                self.file_count += 1
                file_path = os.path.join(save_dir, f"{base_filename}_{self.file_count}")

            Tab_add.ADDTAB.save_text(selected_data_df, file_path)
        except Exception as exc:
            logging.error("保存文件时出错: %s", exc)

    # 处理表格勾选状态变化。
    def check_check_state(self, item):
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

    # 重绘当前选中的传感器曲线。
    def redraw(self):
        for sensor_name in self._data_visible:
            try:
                index = g_var.app_state.sensor_names.index(sensor_name)
                data_list = self.data[index]
                if data_list:
                    if sensor_name in self._data_lines:
                        self._data_lines[sensor_name].setData(range(len(data_list)), data_list)
                    else:
                        pen_color = self.get_currency_color(sensor_name)
                        self._data_lines[sensor_name] = self.plot_widget.plot(
                            range(len(data_list)),
                            data_list,
                            pen=pg.mkPen(pen_color, width=3),
                        )
            except Exception as exc:
                logging.error("更新图表时出错: %s", exc)

        sensors_to_remove = [name for name in self._data_lines if name not in self._data_visible]
        for sensor_name in sensors_to_remove:
            try:
                line = self._data_lines.pop(sensor_name, None)
                if line:
                    self.plot_widget.removeItem(line)
            except Exception as exc:
                logging.error("移除绘图线时出错: %s", exc)

    # 刷新传感器数值表格。
    def update_table(self):
        try:
            if self.model.rowCount() != len(g_var.app_state.sensor_names):
                self.model.setRowCount(len(g_var.app_state.sensor_names))

            for i, sensor_name in enumerate(g_var.app_state.sensor_names):
                value = self.data[i][-1] if self.data[i] else 0
                name_item = self.model.item(i, 0)
                if not name_item:
                    name_item = QStandardItem()
                    name_item.setCheckable(True)
                    name_item.setCheckState(Qt.CheckState.Checked)
                    name_item.setEditable(False)
                    self.model.setItem(i, 0, name_item)

                if name_item.text() != sensor_name:
                    name_item.setText(sensor_name)
                    name_item.setForeground(QBrush(QColor(self.get_currency_color(sensor_name))))

                should_be_checked = sensor_name in self._data_visible
                target_state = Qt.CheckState.Checked if should_be_checked else Qt.CheckState.Unchecked
                if name_item.checkState() != target_state:
                    name_item.setCheckState(target_state)

                value_item = self.model.item(i, 1)
                if not value_item:
                    value_item = QStandardItem()
                    value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    value_item.setEditable(False)
                    self.model.setItem(i, 1, value_item)

                new_value_text = f"{value:.2f}"
                if value_item.text() != new_value_text:
                    value_item.setText(new_value_text)
        except Exception as exc:
            logging.error("更新表格时出错: %s", exc)

    # 生成指定数量的随机颜色列表。
    def generate_random_color_list(self, length):
        return [self.generate_random_hex_color() for _ in range(length)]

    # 获取指定传感器的固定颜色。
    def get_currency_color(self, sensor):
        if sensor not in self._data_colors:
            self._data_colors[sensor] = next(self.color_cycle)
        return self._data_colors[sensor]

    # 生成一个随机十六进制颜色值。
    def generate_random_hex_color(self):
        return "#{:02x}{:02x}{:02x}".format(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    # 页面关闭时释放线程和串口资源。
    def closeEvent(self, event):
        try:
            if hasattr(self, "time_th") and self.time_th:
                self.time_th._running = False
                self.time_th._stop_evt.set()
            if hasattr(self, "Serialopea") and self.Serialopea:
                self.Serialopea._running = False
            if hasattr(self, "serial_controller"):
                self.serial_controller.shutdown()
            time.sleep(0.1)
            event.accept()
        except Exception as exc:
            logging.error("关闭图形页时出错: %s", exc)
            event.accept()

    # 处理绘图控件上的快捷键事件。
    def eventFilter(self, source, event):
        if source is self.plot_widget and event.type() == QEvent.KeyPress:
            if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
                self.savefile()
                return True
        return super().eventFilter(source, event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphShowWindow()
    window.show()
    sys.exit(app.exec())
