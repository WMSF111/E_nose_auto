import os
import threading
from dataclasses import dataclass, field
from typing import List

headers_list = []
textEdit_DataFrame = ' '
textEdit_nolc_DataFrame = ' '
folders = ' '
trainfile_txt_path = ' '
header_trainfile_txt_path = ' '
folder_path = ' '
filter_preprocess = ' '
themeFile = os.path.join(os.path.dirname(__file__), "py_dracula_light.qss")
Com_select = " "
draw_flag = False
lock = threading.Lock()

DEFAULT_PRIMARY_PORT = ""
DEFAULT_SECONDARY_PORT = ""
DEFAULT_PRIMARY_BAUD = 115200
DEFAULT_SECONDARY_BAUD = 9600
DEFAULT_AUTO_MODE = False
DEFAULT_SAMPLE_TIME = 0
DEFAULT_EXHAUST_TIME = 0
DEFAULT_BASE_TIME = 0
DEFAULT_TARGET_TEMP = 0.0
DEFAULT_TARGET_SAMPLE_COUNT = 1
DEFAULT_CURRENT_TEMP = 1.0
DEFAULT_TARGET_TEMP_TIME = 0
DEFAULT_ROOM_TEMP = 1
DEFAULT_CURRENT_CHANNEL = 1
DEFAULT_CURRENT_SAMPLE_INDEX = 0
DEFAULT_SAVE_FLAG = 0

default_sensor_names = [
    "sensor1", "sensor2", "sensor3", "sensor4", "sensor5",
    "sensor6", "sensor7", "sensor8", "sensor9", "sensor10",
    "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
    "sensor16",
]
default_channels = [0, 4, 3, 2, 1, 5, 6, 7, 8]
thepos = [[0, 0, 0], [944, 3184, 15480]]
default_positions = [
    [0, 0, 0],
    [944, 3184, 1548],
    [1536, 3184, 1548],
    [960, 18943, 1548],
    [1728, 19455, 1548],
]


@dataclass
class SerialConfig:
    primary_port: str = DEFAULT_PRIMARY_PORT
    secondary_port: str = DEFAULT_SECONDARY_PORT
    primary_baud: int = DEFAULT_PRIMARY_BAUD
    secondary_baud: int = DEFAULT_SECONDARY_BAUD
    auto_mode: bool = DEFAULT_AUTO_MODE


@dataclass
class ExperimentConfig:
    sample_time: int = DEFAULT_SAMPLE_TIME
    exhaust_time: int = DEFAULT_EXHAUST_TIME
    base_time: int = DEFAULT_BASE_TIME
    target_temp: float = DEFAULT_TARGET_TEMP
    target_sample_count: int = DEFAULT_TARGET_SAMPLE_COUNT


@dataclass
class RuntimeState:
    current_temp: float = DEFAULT_CURRENT_TEMP
    target_temp_time: int = DEFAULT_TARGET_TEMP_TIME
    room_temp: int = DEFAULT_ROOM_TEMP
    current_channel: int = DEFAULT_CURRENT_CHANNEL
    current_sample_index: int = DEFAULT_CURRENT_SAMPLE_INDEX
    save_flag: int = DEFAULT_SAVE_FLAG
    demo_mode: bool = True
    demo_step_seconds: int = 5


@dataclass
class SensorConfig:
    names: List[str] = field(default_factory=lambda: list(default_sensor_names))
    channels: List[int] = field(default_factory=lambda: list(default_channels))
    positions: List[List[int]] = field(default_factory=lambda: [list(item) for item in default_positions])


class AppState:
    # 初始化应用运行时状态对象。
    def __init__(self) -> None:
        self.serial = SerialConfig()
        self.experiment = ExperimentConfig()
        self.runtime = RuntimeState()
        self.sensor_config = SensorConfig()

    # 更新串口端口配置。
    def set_serial_ports(self, primary: str, secondary: str = "") -> None:
        self.serial.primary_port = primary
        self.serial.secondary_port = secondary

    # 更新采样相关时间配置。
    def set_sample_config(self, sample_time: int, exhaust_time: int, base_time: int) -> None:
        self.experiment.sample_time = sample_time
        self.experiment.exhaust_time = exhaust_time
        self.experiment.base_time = base_time

    # 更新是否启用自动模式。
    def set_auto_mode(self, enabled: bool) -> None:
        self.serial.auto_mode = enabled

    # 更新传感器名称列表。
    def update_sensor_names(self, sensor_names: List[str]) -> None:
        self.sensor_config.names = list(sensor_names)

    # 更新当前温度值。
    def update_current_temp(self, current_temp_value: float) -> None:
        self.runtime.current_temp = current_temp_value

    # 更新目标温度值。
    def set_target_temp(self, target_temp_value: float) -> None:
        self.experiment.target_temp = target_temp_value

    # 更新目标采样数量。
    def set_target_sample_count(self, count: int) -> None:
        self.experiment.target_sample_count = count

    # 设置是否启用演示模式。
    def set_demo_mode(self, enabled: bool) -> None:
        self.runtime.demo_mode = enabled

    # 设置演示模式下每步等待秒数。
    def set_demo_step_seconds(self, seconds: int) -> None:
        self.runtime.demo_step_seconds = max(1, int(seconds))

    # 返回当前是否为自动模式。
    @property
    def auto_mode(self) -> bool:
        return self.serial.auto_mode

    # 返回当前传感器名称列表。
    @property
    def sensor_names(self) -> List[str]:
        return self.sensor_config.names

    # 设置当前传感器名称列表。
    @sensor_names.setter
    def sensor_names(self, names: List[str]) -> None:
        self.sensor_config.names = list(names)

    @property
    def primary_port(self) -> str:
        return self.serial.primary_port

    @property
    def secondary_port(self) -> str:
        return self.serial.secondary_port

    @property
    def primary_baud(self) -> int:
        return self.serial.primary_baud

    @property
    def secondary_baud(self) -> int:
        return self.serial.secondary_baud

    @property
    def current_temp(self) -> float:
        return self.runtime.current_temp

    @property
    def target_temp(self) -> float:
        return self.experiment.target_temp

    @property
    def target_sample_count(self) -> int:
        return self.experiment.target_sample_count

    @property
    def sample_time_value(self) -> int:
        return self.experiment.sample_time

    @property
    def exhaust_time_value(self) -> int:
        return self.experiment.exhaust_time

    @property
    def base_time_value(self) -> int:
        return self.experiment.base_time

    @property
    def current_sample_index(self) -> int:
        return self.runtime.current_sample_index

    @current_sample_index.setter
    def current_sample_index(self, value: int) -> None:
        self.runtime.current_sample_index = value

    @property
    def channels(self) -> List[int]:
        return self.sensor_config.channels

    @property
    def positions(self) -> List[List[int]]:
        return self.sensor_config.positions

    @property
    def demo_mode(self) -> bool:
        return self.runtime.demo_mode

    @property
    def demo_step_seconds(self) -> int:
        return self.runtime.demo_step_seconds


app_state = AppState()
