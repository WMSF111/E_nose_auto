'''
帧数据结构创建
'''
import struct


class FrameData():
    def __init__(self, Headflag = 0x55AA, pkgLen = 10, end_num = 0x0A):

        self.Headflag = Headflag
        self.end_num = end_num

        self.pkgLen = pkgLen

        self.buf = ['00' for i in range(self.pkgLen)]
        # 数据头尾定义
        # self.buf[0] = f"{(self.Headflag >> 8) & 0xff:02x}"
        # self.buf[1] = f"{self.Headflag & 0xff:02x}"
        # self.buf[self.pkgLen - 1] = f"{self.end_num:02x}"
        self.buf[0] = '55'
        self.buf[1] = 'AA'
        self.buf[self.pkgLen - 1] = '0A'



    # 将数据包中的数据部分全部设置为0，并更新校验和。
    def setDataToOff(self):
        for i in range(3, self.pkgLen - 1):
            self.buf[i] = '00'

    # 将数据包中的数据部分全部设置为255，并更新校验和。
    def setDataToOn(self):
        for i in range(2, self.pkgLen - 1):
            self.buf[i] = 'FF'

    def setDataToArray(self, arr): # 填充数据位
        alen = len(arr)
        dlen = self.pkgLen - 3
        rlen = alen if (alen < dlen) else dlen
        for i in range(2, rlen + 2):
            self.buf[i] = arr[i - 2]


    def setDataTodo(self, opea, opea1 = 0, opea2 = 0, opea3 = 0):
        self.setDataToOff()
        if(opea != "0A" and opea != "0B" and opea != "0C" and opea != "0D"):
            opea = f"{opea:02x}"
        self.buf[2] = opea
        # print("setDataTodo:", opea, opea1, opea2)
        if (opea == '01'): # 加热xx(01-08)通道到目标温度yyyy(00C8---0384)
            self.buf[3] = f"{opea1 & 0xff:02x}"
            opea2 = int(10 * opea2)
            hex_string = f"{opea2:04x}"
            self.buf[4] = hex_string[:2]
            self.buf[5] = hex_string[-2:]
        if (opea == '02'): # 读取xx(01-08)通道的实时温度
            self.buf[3] = f"{opea1 & 0xff:02x}"
        if (opea == '03' or opea == '0B' or opea == '0C' or opea == '0D'): # 吸取xx毫升的气体，用时yy秒。
            # 0B : 运读取当前坐标，返回55 AA 0B XX XX  YY YY  ZZ  ZZ  0A
            # OC : 运动轴回到初始0坐标点,校准坐标轴。
            # 0D : 取初始化标识,返回55 AA 0D 00  xx  00  yy  00  zz  0A
            opea1 = 1
        if (opea == '04'): # xxxx表示清洗时长，单位秒。
            hex_string = f"{opea1:04x}"
            self.buf[3] = hex_string[:2]
            self.buf[4] = hex_string[-2:]
        if (opea == '05'): # 00：关闭气泵 01：打开气泵
            self.buf[3] = f"{opea1 & 0xff:02x}"
        if(opea == '0A'): # 运动到指定位置位置（XXXX,YYYY,ZZZZ）
            hex_string = f"{opea1:04x}"
            self.buf[3] = hex_string[:2]
            self.buf[4] = hex_string[-2:]
            hex_string = f"{opea2:04x}"
            self.buf[5] = hex_string[:2]
            self.buf[6] = hex_string[-2:]
            hex_string = f"{opea3:04x}"
            self.buf[7] = hex_string[:2]
            self.buf[8] = hex_string[-2:]
        return self.buf


    def packBytes(self, printf = False):
        if printf == True:
            print("packBytes:",self.buf)
        hex_string = ' '.join(self.buf)
        byte_data = bytes.fromhex(hex_string.replace(' ', ''))
        return byte_data


# '''
# 帧数据结构解析
# '''
#
# from event import *
#
#
# class FramePkg(EventDispatcher):
#     def __init__(self, pixStyle=0x14, width=1, height=1):
#         EventDispatcher.__init__(self)
#         self.pixStyle = pixStyle
#         self.frameWidth = width
#         self.frameHeight = height
#
#         self.pkgLen = self.frameWidth * self.frameHeight * (pixStyle & 0x0f) + 6
#         self.bufLen = self.pkgLen * 3
#         self.buf = [0 for i in range(self.bufLen)]
#         self.dataLen = self.pkgLen - 6
#         self.linedata = [0 for i in range(self.dataLen)]
#         self.startIdx = 0
#         self.endIdx = 0
#
#     # 获取数据长度 int
#     def getDatLen(self):
#         if (self.endIdx >= self.startIdx):
#             return self.endIdx - self.startIdx
#         return self.bufLen + self.endIdx - self.startIdx
#
#     # 判断数据长度已经到达一个包长bool
#     def pkgIsOk(self):
#         return self.getDatLen() >= self.pkgLen
#
#     # 检测包头为0xff并且有一个包长的数据bool
#     def chkStart(self):
#         while (self.buf[self.startIdx] != 0xff and self.pkgIsOk()):
#             self.nextByte()
#         return self.pkgIsOk()
#
#     # 检测索引超出buf末尾，从头开始
#     def chkMaxIdx(self, idx):
#         if (idx > self.bufLen - 1):
#             idx %= self.bufLen  # ??
#         return idx
#
#     # 获取包尾索引int
#     def getPkgEndIdx(self):
#         return self.chkMaxIdx(self.startIdx + self.pkgLen - 1)
#
#     # 检查包尾为0xfe bool
#     def chkEnd(self):
#         return self.buf[self.getPkgEndIdx()] == 0xfe
#
#     # 相对起始索引包数据的每个字节索引的映射int
#     def getIdxFromStart(self, i):
#         return self.chkMaxIdx(self.startIdx + i)
#
#     # 检测校验和bool
#     def chkCrc(self):
#         crcidx = self.getPkgEndIdx() - 1  # 包尾之前的一个字节
#         if (crcidx < 0):
#             crcidx = self.bufLen + crcidx  # 包尾索引在缓冲区前面，包头在后面
#         d = self.buf[crcidx]
#         # print(d)
#         r = 0
#         # 计算和
#         for i in range(1, self.pkgLen - 2):
#             # print(self.getIdxFromStart(i),":",self.buf[self.getIdxFromStart(i)])
#             r += self.buf[self.getIdxFromStart(i)]
#         r = r & 0xff
#         # print(r)
#         return d == r
#
#     # 写数据到显示区
#     def writeData(self):
#         r = 0
#         for i in range(4, self.pkgLen - 2):
#             r = self.buf[self.getIdxFromStart(i)]
#             # print(r)
#             self.linedata[i - 4] = r
#
#     def nextPkg(self):
#         self.startIdx = self.chkMaxIdx(self.getPkgEndIdx() + 1)
#
#     def nextByte(self):
#         self.startIdx = self.chkMaxIdx(self.startIdx + 1)
#
#     # 解析包
#     def parsePKG(self):
#         while (self.pkgIsOk()):
#             if (self.chkStart() and self.chkEnd() and self.chkCrc()):
#                 self.writeData()
#                 self.dispatch_event(PKGEvent(PKGEvent.PKG_DATA_OK, self.linedata))
#                 self.nextPkg()
#             else:
#                 self.nextByte()
#
#     def writeToBuf(self, byt):
#         self.buf[self.endIdx] = byt
#         self.endIdx = self.chkMaxIdx(self.endIdx + 1)
#
#     def writeArrayToBuf(self, arr):
#         for i in range(len(arr)):
#             self.writeToBuf(arr[i])
#         self.parsePKG()
#     # '''

#
# if __name__ == '__main__':
#     d = FrameData(0x14, 10, 2)
#
#     print("init", d.buf)
#     d.setDataToOn()
#     print("on", d.buf)
#     d.setDataToRGBW(255)
#     print("r", d.buf)
#     d.setDataToRGBW(0, 255)
#     print("g", d.buf)
#     d.setDataToRGBW(0, 0, 255)
#     print("b", d.buf)
#     d.setDataToRGBW(0, 0, 0, 255)
#     print("w", d.buf)
#     d.setDataToRGBW(255, 255, 255, 255)
#     print("rgbw", d.buf)
#     d.setDataToArray([0, 0, 0])
#     print("000", d.buf)
#     import time
#
#     while True:
#         time.sleep(1)
