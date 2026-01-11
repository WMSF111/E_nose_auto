import threading, os

headers_list = [] # 列头列表
textEdit_DataFrame = ' ' # 编辑框的文字
textEdit_nolc_DataFrame = ' ' # 编辑框内无列头行头数据
folders = ' '
trainfile_txt_path = ' ' # 训练集路径
header_trainfile_txt_path = ' ' # 训练集路径
folder_path = ' '
# 算法
filter_preprocess = ' '
themeFile = os.path.join(os.path.dirname(__file__), "py_dracula_light.qss")


Com_select = " "
Port_select = ""
Port_select2 = ""
Bund_select2 = 9600
Bund_select = 115200
now_temp = 1.0
target_temp = 0
target_temp_time = 0
room_temp = 1
channal = [0, 4,3,2,1,5,6,7,8]
draw_flag = False
now_chan = 1
now_Sam = 0
target_Sam = 1
sample_time = 0
exhaust_time = 0
base_time = 0
lock = threading.Lock()  # 创建一个锁
Save_flag = 0
Auto_falg = False

sensors = [
    "sensor1", "sensor2", "sensor3", "sensor4", "sensor5",
    "sensor6", "sensor7", "sensor8", "sensor9", "sensor10",
    "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
    "sensor16"
]

thepos = [[0,0,0],
    [944, 3184, 15480]]
posxyz = [
    [0,0,0],
    [944, 3184, 1548],
    [1536, 3184, 1548],
    [960, 18943, 1548],
    [1728, 19455, 1548],
]