import serial
import serial.tools.list_ports
import threading
import tool.frame_data as FrameDate


# 获取系统中所有可用的串口列表
def getPortList():
    Com_Dict = {}
    port_list = list(serial.tools.list_ports.comports())
    for port in port_list:
        Com_Dict["%s" % port[0]] = "%s" % port[1]
    return port_list, Com_Dict


# 自定义串口类
class myserial():
    def __init__(self, port="", bund=0, hex_flag = 0):
        # 初始化串口类
        self.read_flag = False  # 读取标志，用于控制读取线程

        self.sendSignal = None
        self.getSignal = None
        self.sameSignal = False

        self.pause_flag = False  # 暂停标志，用于控制暂停和恢复
        self.busy = False
        self.hex_flag = hex_flag # 用于判断是不是16进制读取
        self.port = port  # 串口名称
        self.bund = bund  # 波特率
        self.lock = threading.Lock()  # 创建一个锁
        self.ser = None  # 初始化串口对象
        self.d = FrameDate.FrameData()

    def setSer(self, port: str, bund: int):
        # 设置串口名称和波特率
        self.port = port
        self.bund = bund

    def open(self, fun, slip = None, flag = 0, stock = 0, timeout=1000):
        # flag == 1, 16进制
        self.hex_flag = flag
        # 打开串口
        try:
            # 创建串口对象
            if stock == 0:
                self.ser = serial.Serial(self.port, self.bund, timeout=timeout)
            else:
                self.ser = serial.Serial(self.port, self.bund)
            if self.ser.is_open:
                print("创建串口成功")
                # 如果串口成功打开
                with self.lock:
                    self.read_flag = True  # 设置读取标志为 True
                    self.pause_flag = False  # 设置暂停标志为 False
                rt = threading.Thread(target=self.loopRead, args=(fun,slip))  # 创建读取线程
                # rt.setDaemon(True)  # 设置为守护线程，确保主线程结束时子线程也会结束
                # rt.setTerminationEnabled(True)
                rt.start()  # 启动读取线程

                return 0, "打开串口%s成功,波特率%d\n" % (self.port, self.bund)  # 返回成功信息
        except Exception as e:
            # 如果打开串口失败
            with self.lock:
                self.read_flag = False  # 设置读取标志为 False
            return 1, "打开串口%s失败\n%s\n" % (self.port, str(e))  # 返回失败信息

    def write(self, text):
        # 向串口写入数据
        if self.ser:
            if self.hex_flag == 0:
                result = self.ser.write(text.encode('utf-8'))  # 将字符串编码为字节并写入串口
                self.sendSignal = text.strip()
            else:
                # 转换为大写并用空格连接
                self.sendSignal = ' '.join(b.upper() for b in text[:])
                self.sendSignal = self.sendSignal.strip()  # 去除首尾的空白字符、换行符、回车符
                text = self.hexarrytobytes(text) # hex
                result = self.ser.write(text)
            print("self.sendSignal: ", self.sendSignal)
            self.sameSignal = False
            return result  # 返回写入的字节数
        else:
            return 0

    def serialSend(self, opea, opea1 = 0, opea2 = 0, opea3 = 0, flag = False, re = None): # 发送 FrameData 对象中的数据到串口。
        if not self.busy:
            if hasattr(self, 'ser'):
                try:
                    self.busy = True
                    text = self.d.setDataTodo(opea, opea1, opea2, opea3)  # 切换到下一个样品位置
                    if (flag): print(text)
                    self.write(text)
                    if re != None:
                        self.sendSignal = re
                    self.busy = False
                except serial.serialutil.SerialException:
                    self.no_error = False

    def loopRead(self, fun, slip):
        """
        自动识别数据类型：
          - 纯可打印 ASCII → utf-8 文本
          - 含非打印或 0x00 → 十六进制字符串
        fun(line: str) 回调收到的内容
        """
        buffer = bytearray()  # 创建可变字节数组
        slip_n = b'\n\r'  # 用于按行切帧
        if slip != None:
            slip_n = slip.encode()  # 将分隔符转换为字节串

        while True:
            with self.lock:
                if not self.read_flag:
                    break
                if self.pause_flag:
                    continue

            if self.ser.in_waiting:
                data = self.ser.read(self.ser.in_waiting)
                buffer.extend(data)
                # 判断是否是十六进制接收模式
                if self.hex_flag == 0:
                    # 判断缓冲区内是否有完整帧
                    while slip_n in buffer:
                        frame, buffer = buffer.split(slip_n, 1)  # 按 \r\n 切帧
                        # 处理 ASCII 数据帧
                        text = frame.decode('utf-8', errors='replace')
                        # 回调
                        fun(text)
                        text = text.strip()
                        if (text and
                                text[0] == "1" or text[0] == "2" or text[0] == "3" or text[0] == "4"or text[0] == "0"):
                            # 打印当前接收到的信号与预期信号的比较
                            self.getSignal = text.strip()
                            print("self.getSignal: ", self.getSignal)
                            self.sameSignal = (self.getSignal == self.sendSignal)
                            # print("是否相等：", self.sameSignal)
                else:  # 十六进制接收模式
                    # 查找 '55 AA' 在字节数组中的位置
                    start_index = buffer.find(b'\x55\xAA')

                    if start_index != -1:
                        # 确保帧头之后，接收到至少10字节的数据
                        if len(buffer) >= start_index + 10:
                            # 提取完整的数据帧
                            frame = buffer[start_index:start_index + 10]

                            # 为每个字节生成十六进制字符串
                            hex_data = ' '.join(f'{b:02X}' for b in frame)
                            text = hex_data

                            # 打印当前接收到的信号与预期信号的比较
                            self.getSignal = text.strip()
                            # print("self.getSignal: ", self.getSignal)

                            # 比较接收到的信号与发送信号是否一致
                            self.sameSignal = (self.getSignal == self.sendSignal)
                            # print("是否相等：", self.sameSignal)

                            # 回调
                            fun(text)

                            # 清空缓冲区，避免重复处理
                            buffer = buffer[start_index + 10:]




    def pause(self):
        # 暂停串口通信
        with self.lock:
            self.pause_flag = True  # 设置暂停标志为 True
        print("暂停串口通信")

    def resume(self):
        # 恢复串口通信
        with self.lock:
            self.pause_flag = False  # 设置暂停标志为 False
        print("恢复串口通信")

    def stop(self):
        # 停止串口通信
        with self.lock:
            self.read_flag = False  # 设置读取标志为 False
        if self.ser:
            self.ser.close()  # 关闭串口
        print("关闭串口")

    def setALLDataToArray(self, arr):
        for i in range(0, self.pkgLen):
            self.buf[i] = arr[i]

    def hexarrytobytes(self, dat): # 发送数据到串口，先将数据打包成字节串，然后写入串口。
        req = ' '.join(dat) # 数组转字符串
        # 将字符串 req 中的十六进制表示的内容转换为字节（bytes）类型。
        req = bytes.fromhex(req.replace(' ', ''))
        return req





class SerialsMng(): # 管理多线程
    # 接收一个列表 lst，列表中包含串口的配置信息
    # list=[name,bps,pixStyle,width,  name,bps,pixStyle,width,]
    # ["COM3",250000,0x13, 45,"COM4",250000,0x13, 45]
    def __init__(self, lst):

        self.ser_count = int(len(lst) / 2) # 计算串口设备的数量，每个设备占用 4 个配置项。
        self.ser_arr = [] # 初始化一个空列表，用于存储串口设备对象。
        for x in range(0, self.ser_count):
            # 遍历配置列表，为每个串口设备创建一个 SerialOP 对象，并将其添加到 self.ser_arr 列表中。
            idx = x * 2
            sop = myserial(lst[idx], lst[idx + 1])
            print(lst[idx], lst[idx + 1])
            self.ser_arr.append(sop)
        print(self.ser_arr) # 打印串口设备对象列表。



    def splitData(self, data, sidx=0, eidx=0): # 根据起始索引和结束索引，从数据列表中截取子列表。
        b = eidx <= len(data) and sidx >= 0 and eidx - sidx > 0
        if (b):
            d = data[sidx:eidx]
            return b, d
        return False, []

    def sendFrameData(self, data, pixstyle=4, width=10): # 将数据分割并发送到每个串口设备。
        datlen = len(self.ser_arr)
        # 每个串口设备控制一行
        c = 0
        u = pixstyle * width
        # print(u)
        for x in range(datlen):  #

            # 获取切割的数据，发送到对应的节点
            b, d = self.splitData(data, c * u, (c + 1) * u)
            # print(c, c * u, (c + 1) * u,b,d)
            if b:
                # 串口发送1

                self.setdataAndsend(x, d)
            c += 1

    def sendFrameData_splited(self, data): #如果数据已经分割好，直接将每个子列表发送到对应的串口设备。
        dvclen = len(self.ser_arr)

        datlen = len(data)
        # 每个串口设备控制 data 一个子数组的数据
        if dvclen < datlen:
            datlen = dvclen
        # print(u)
        for x in range(datlen):  #
            self.setdataAndsend(x, data[x])
