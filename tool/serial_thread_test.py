# -*- coding: utf-8 -*-
import serial, time, threading, struct
import Enose.tool.serial_thread as ser

# pip install pyserial
class SerialsMng(): # 管理多线程
    # 接收一个列表 lst，列表中包含串口的配置信息
    # list=[name,bps,pixStyle,width,  name,bps,pixStyle,width,]
    # ["COM3",250000,0x13, 45,"COM4",250000,0x13, 45]
    def __init__(self, lst):

        self.ser_count = int(len(lst) / 4) # 计算串口设备的数量，每个设备占用 4 个配置项。
        self.ser_arr = [] # 初始化一个空列表，用于存储串口设备对象。
        for x in range(0, self.ser_count):
            # 遍历配置列表，为每个串口设备创建一个 SerialOP 对象，并将其添加到 self.ser_arr 列表中。
            idx = x * 2
            sop = SerialOP(lst[idx], lst[idx + 1])
            print(lst[idx], lst[idx + 1])
            self.ser_arr.append(sop)
        print(self.ser_arr) # 打印串口设备对象列表。

    def setdataAndsend(self, idx, data): # 设置指定串口设备的数据并发送。

        sop = self.ser_arr[idx]
        # print("xxxxxxxxxxxxxxxxxxx")
        # print( data)
        # print(sop.d.pkgLen)
        if not sop.busy:
            sop.d.setDataToArray(data)
            # print(data)
            # print(sop.d.buf)

            sop.serialSend()


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


from Enose.tool.frame_data import FrameData
import struct
from ctypes import create_string_buffer


class SerialOP():
    no_error = True

    def __init__(self, serialPort, baudRate):
        # 构造函数，打开串口，创建 FrameData 对象，并启动自动重连线程。
        # serialPort = "COM3"  # 串口
        # baudRate =250000  # 波特率
        self.serialPort = serialPort
        self.busy = False
        self.baudRate = baudRate
        self.createSer()
        self.d = FrameData()
        t = threading.Thread(target=self.thread_autoReCreat)  # 自动连接串口线程
        t.daemon = True
        t.start()

    def createSer(self):# 尝试打开串口，如果成功则设置标志位，否则打印错误信息。
        try:
            self.ser = serial.Serial(self.serialPort, self.baudRate, timeout=0)  # !!!!!!!!!!!!!!!!!!!!!!!!!!无阻塞
            self.no_error = True
            print("参数设置：串口=%s ，波特率=%d" % (self.serialPort, self.baudRate))
            self.busy = False
        except:
            self.no_error = False
            print("ERROE:参数设置：串口=%s ，波特率=%d" % (self.serialPort, self.baudRate))

    def thread_autoReCreat(self): # 自动重连线程，如果串口连接失败，每隔 1 秒尝试重新连接。
        while 1:
            if (not self.no_error):
                print("serail err relinking..")
                try:
                    self.createSer()
                except:
                    pass
            else:
                pass
            time.sleep(1)

    def serialSendData(self, dat): # 发送数据到串口，先将数据打包成字节串，然后写入串口。
        # datlen = len(dat)
        # packstyle = str(datlen) + 'B'  # B 0-255
        # req = struct.pack(packstyle, *dat)
        req = ' '.join(dat)
        req = bytes.fromhex(req.replace(' ', ''))
        if hasattr(self, 'ser'):
            try:
                self.ser.write(req)
            except serial.serialutil.SerialException:
                self.no_error = False

    def serialSend(self): # 发送 FrameData 对象中的数据到串口。
        if not self.busy:
            if hasattr(self, 'ser'):
                # print("serialSend")
                try:
                    self.busy = True
                    self.ser.write(self.d.packBytes())
                    self.busy = False
                    # print( self.ser.readline())#read会阻塞
                except serial.serialutil.SerialException:
                    self.no_error = False

    def testCreateDataToSend(self):
        slp = 1
        self.d.setDataToOn()
        print("on", self.d.buf)
        # self.d.packBytes()
        self.serialSendData(self.d.buf)
        time.sleep(slp)
        self.d.setDataTodo(1,2,100)
        print("1:", self.d.buf)
        self.serialSendData(self.d.buf)
        time.sleep(slp)
        self.d.setDataTodo(2,2)
        print("2:", self.d.buf)
        self.serialSendData(self.d.buf)
        time.sleep(slp)
        self.d.setDataTodo(2,100)
        print("3:", self.d.buf)
        self.serialSendData(self.d.buf)
        time.sleep(slp)


    def thread_ssend(self):
        # 收发数据
        while 1:
            self.testCreateDataToSend()

            if (not self.no_error):
                print("serail err relinking..")
                try:
                    self.createSer()
                except:
                    pass
                time.sleep(1)

    def thread_srecv(self):
        while True:
            line = self.ser.readline()  # 只读一次
            if line:  # 有数据才处理
                print("recv:", line)  # 打印原始字节
                print("\n")


if __name__ == "__main__":
    def main():
        # client socket
        # sop = SerialOP("COM1", 115200, 0x13, 10)
        # 分别启动听和说线程
        # t = threading.Thread(target=sop1.thread_ssend)  # 注意当元组中只有一个元素的时候需要这样写, 否则会被认为是其原来的类型
        # t.daemon=True
        # t.start()
        # sop2 = SerialOP("COM4", 115200, 0x13, 10)
        # t1 = threading.Thread(target=sop2.thread_srecv)
        # t1.daemon=True
        # t1.start()
        import time

        while 1:
            # sop.testCreateDataToSend()
            time.sleep(1)


    # main()
    sconfig = ["COM1", 115200, "COM3", 9600, "COM4", 9600]  #
    # sconfig = ["COM1", 115200, 0x13, 45]  #
    smng = SerialsMng(sconfig)
    # smng.ser_arr[0].open()
    t1 = threading.Thread(target=smng.ser_arr[0].thread_srecv)
    t1.daemon=True
    t1.start()
    t2 = threading.Thread(target=smng.ser_arr[2].thread_srecv)
    t2.daemon = True
    t2.start()
    while 1:
        smng.setdataAndsend(1, ['FF', '00', '00', 'FF', '00', '00'])
        time.sleep(1)
        smng.setdataAndsend(1, ['FF', 'F0', '0F', 'FF', 'B0', '0B'])
        time.sleep(1)
#         # smng.setdataAndsend(0, [0, 0, 0xff, 0, 0, 0xff])
#         # time.sleep(1)
#         # smng.setdataAndsend(0, [0xff, 0xff, 0xff, 0xff, 0xff, 0xff])  # '''
#         # time.sleep(1)
#         # print('123')
#     # ser.close()