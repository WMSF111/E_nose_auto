import tool.frame_data as frame_data
import threading
import time, copy
import global_var as glo_var

class time_thread(): # 时间相关的线程
    # 初始化输入两个串口， 停止时间与初始时间
    def __init__(self,  ui = None, stoptime = 0, starttime = 0):
        self.time = starttime
        self.ms = ui
        self.stoptime = stoptime
        self.lock = threading.Lock()  # 创建一个锁
        self._stop_evt = threading.Event()  # 用来打断 sleep
        self._running = True
        self.opea = None # 串口操作
        self.timeout = 1000; # 循环时间

    def thread_loopfun(self, fun, timeout=1000): # 输入要循环的函数和循环时间
        # self.time = 0
        try:
            with self.lock:
                self.timeout = timeout
                self._running = True
                self._thread = threading.Thread(
                    target=fun,
                    daemon=True # 当主线程退出时，所有守护线程也会随之被终止
                )
                self._thread.start()
                return 0, f"开始计时，每{timeout}ms触发一次\n"
        except Exception as e:
            return 1, f"计时失败：{e}\n"

    def run_time(self): # 每1s读取一次温度，直到达到合适温度glo_var.target_temp
        while self._running:
            self.time += 1
            time.sleep(self.timeout / 1000)

    def thread_draw(self, fun, timeout=1000): # 输入要调用循环函数返回结果的函数fun和循环时间
        try:
            with self.lock:
                self.timeout = timeout
                self._running = True
                self._thread = threading.Thread(
                    target=self.loopTime_arri,
                    args=(fun,),
                    daemon=True
                )
                self._thread.start()
                return 0, f"开始计时，每{timeout}ms触发一次\n"
        except Exception as e:
            return 1, f"计时失败：{e}\n"

    def loopTime_arri(self, fun): # 不断累加时间，不用管和上面是一起用的
        while True:
            if self._running:
                # 线程安全：把值读出来再比较
                self.time += 1
                fun(self.time) # fun函数调用累加的时间
                time.sleep(self.timeout / 1000)
                if(self.stoptime != 0 and self.time == self.stoptime):
                    self.ms._Clear_Button.emit(True) # 允许继续
                    self._running =  False
            if self._stop_evt.is_set():
                print("Thread stopped by event.")
                break

    # —— 供外部调用的停止接口 ——
    def pause(self, name = None):
        with self.lock:  # 若存在并发访问，加锁
            self._running = False
        if name != None:
            print(name + "线程暂停")
        else:
            print("时间线程暂停")

    def resume(self, name = None):
        with self.lock:
            self._running = True  # 设置暂停标志为 False
        if name != None:
            print(name + "恢复时间线程")
        else:
            print("恢复时间线程")

    def stop(self, name = None):
        glo_var.now_temp = 0
        self._running = False  # 设置读取标志为 False
        self.time = 0
        self._stop_evt.set()  # 通过事件通知线程停止
        if name != None:
            print(name + "线程已结束")
        else:
            print("线程已结束")

channal = [0, 4,3,2,1,5,6,7,8]
posxyz = [
    [0,0,0],
    [944, 3184, 1548],
    [1536, 3184, 1548],
    [960, 18943, 1548],
    [1728, 19455, 1548],
]
class Serial1opea():
    # 一系列串口控制硬件的操作
    def __init__(self, ms, ser, ser1 = None):
        self.ser = ser
        self.ser1 = ser1
        self.ms = ms
        # self.time_th = None
        self._running =  True

    def GetSigal1(self, text): # ser1获取信号
        parts = text.split()  # ['55','AA', ...]
        Frame = frame_data.FrameData()
        parts = parts[2 : Frame.pkgLen - 1]
        Frame.setDataToArray(parts)
        if (Frame.buf[2] == '02'): # 获取温度
            num = int.from_bytes(bytes.fromhex(Frame.buf[4] + Frame.buf[5]), byteorder='big')  # 1000
            text = num / 10.0
            glo_var.now_temp = text # 更新现在的温度

    def loop_to_target_temp(self, the_time = 0): # 每1s读取一次温度，直到达到合适温度glo_var.target_temp
        while True:  # 没到回复
            if self._running == False:
                break
            # 线程安全：把值读出来再比较
            print("现在温度是：", glo_var.now_temp)
            self.ms._Currtem_spinBox.emit(glo_var.now_temp)
            self.ser1.serialSend(2, glo_var.channal[1])
            the_time += 1
            if glo_var.target_temp <= glo_var.now_temp:  # 当目标温度达成
                print("达到目标温度需要时间：",the_time)
                self.ms._attendtime_spinBox.emit(the_time)
                break
            time.sleep(1)

    def autosample(self):
        self.base_clear()
        self.loop_to_target_temp()# 循环到达指定温度
        self.now_Sam = 1
        while self.now_Sam <= glo_var.target_Sam and self._running == True:  # target_Sam个样品循环操作
            print("开始操作:self.now_Sam", self.now_Sam)
            time.sleep(4)

            self.pos_down()  # 插入
            if self._running == False:
                break
            time.sleep(4)
            print("样品" + str(self.now_Sam) + "开始采集")
            self.sample_collect()  # 采集
            if self._running == False:
                break
            print("气室正在清洗, 下一个为样品" + str(self.now_Sam + 1))
            self.room_clear()
            if self._running == False:
                break
            self.pos_top_before() #拔出当前样品
            time.sleep(2)

            self.pos_top()  # 转移位置到下一个样品
            self.now_Sam += 1

    def auto(self):
        print("自动运行")
        self.base_clear()
        self.sample_collect()
        self.room_clear()

    def base_clear(self):
        if self._running == True:
            self.ms._Clear_Button.emit(False)  # 不允许继续
        text = ("11\n\r")
        num = 0
        while True:  # 没到回复
            if self._running == False:
                break
            self.ser.write(text)  # 开始采集信号
            time.sleep(2)
            num += 1
            if self.ser.sameSignal == True:
                self.ms._statues_label.emit("开始进行基线处理")
                break
            if num == 5:
                self.ms._statues_label.emit("信号串口掉线")
                self.ms._show_error_message.emit("信号串口掉线, 请重新操作")
                self.stop()
                break
        num = 0
        while True:  # 收到回复
            if self._running == False:
                break
            time.sleep(1)
            num += 1
            if self.ser.getSignal == "12":
                self.ms._statues_label.emit("基线处理完成")
                print("基线处理完成")
                self.ms._print.emit(num)
                break
            if num >= glo_var.base_time + 5:
                self.ms._show_error_message.emit(str("基线处理时长超时", num, ", 请重新操作"))
                self.stop()
                break
        self.ms._Clear_Button.emit(True)  # 允许继续

    def sample_collect(self): # 信号采集
        if self._running == True:
            self.ms._Collectbegin_Button.emit(False)

        #开始采集
        text = ("21\n\r")
        num = 0
        self.ms._ClearDraw.emit()  # 清除绘图界面
        if glo_var.Auto_falg == True:
            self.ser_opea(3, glo_var.sample_time, re="55 AA 03 00 01 00 00 00 00 0A") # 信号采集
        while True:  # 没到回复
            if self._running == False:
                break
            self.ser.write(text)  # 开始采集信号
            time.sleep(2)
            num += 1
            if self.ser.sameSignal == True:
                self.ms._statues_label.emit("样品开始采集")
                self.ms._draw_open.emit()
                break
            if num == 5:
                self.ms._statues_label.emit("信号串口掉线")
                self.ms._draw_close.emit()
                self.ms._show_error_message.emit("信号串口掉线, 请重新操作")
                self.stop()
                break
        num = 0
        while True:  # 收到回复
            if self._running == False:
                break
            time.sleep(1)
            num += 1
            if self.ser.getSignal == "22":
                self.ms._statues_label.emit("采样处理完成")
                self.ms._draw_close.emit()
                break
            if num >= glo_var.sample_time + 5:
                self.ms._statues_label.emit("采样时长超时")
                self.ms._draw_close.emit()
                self.ms._show_error_message.emit(str("采样时长超时", num, ", 请重新操作"))
                self.stop()
                break
        self.ms._Collectbegin_Button.emit(True)


    def room_clear(self):
        if self._running == True:
            self.ms._Clearroom_Button.emit(False)
        text = ("31\n\r")
        num = 0
        if glo_var.Auto_falg == True:
            self.ser_opea(4, glo_var.exhaust_time)
        while True:  # 没到回复
            if self._running == False:
                break
            self.ser.write(text)  # 开始采集信号
            time.sleep(2)
            num += 1
            if self.ser.sameSignal == True:
                self.ms._statues_label.emit("气室正在清洗")
                break
            if num == 5:
                self.ms._statues_label.emit("信号串口掉线")
                self.ms._show_error_message.emit("信号串口掉线, 请重新操作")
                self.stop()
                break
        num = 0
        while True:  # 收到回复
            if self._running == False:
                break
            time.sleep(1)
            num += 1
            if self.ser.getSignal == "32":
                self.ms._statues_label.emit("洗气处理完成")
                self.ms._print.emit(glo_var.sample_time + num)
                break
            if num >= glo_var.exhaust_time + 5:
                self.ms._statues_label.emit("洗气时长超时")
                self.ms._Clearroom_Button.emit(True)
                self.ms._show_error_message.emit(str("洗气时长超时：", num, ", 请重新操作"))
                self.stop()
                break
        self.ms._Clearroom_Button.emit(True)

    def stop(self, name = "Serial1opea"):
        self._running = False  # 设置读取标志为 False
        if name != None:
            print(name + "线程已结束")
        else:
            print("线程已结束")

    def pos_top(self):
        if(self.now_Sam != glo_var.target_Sam):
            print("转移位置到下一个样品" + str(self.now_Sam + 1))
            self.ser_opea("0A", posxyz[self.now_Sam + 1][0], posxyz[self.now_Sam + 1][1],
                                (int)(posxyz[self.now_Sam + 1][2] * 0.1), target = 10)
        else:
            self.Stra()
            self.ms._Collectbegin_Button.emit(True)

    def pos_top_before(self):
        print("拔出样品" + str(self.now_Sam))
        self.ser_opea("0A", posxyz[self.now_Sam][0], posxyz[self.now_Sam][1],
                                (int)(posxyz[self.now_Sam][2] * 0.1), target = 10)

    def pos_down(self):
        self.ser_opea("0A", posxyz[self.now_Sam][0], posxyz[self.now_Sam][1],
                                (int)(posxyz[self.now_Sam][2]), target = 20)
        print("插入样品" + str(self.now_Sam))

    def Stra(self): # 运动轴回到原点
        self.ser_opea('0C')

    def ser_opea(self, opea, opea1 = 0, opea2 = 0, opea3 = 0, re = None, target = 5, statues_label = "处理完成"): # ser与ser1发送信号
        num = 0
        while True:  # ser1串口输入
            if self._running == False:
                break
            self.ser1.serialSend(opea, opea1, opea2, opea3, re = re)
            time.sleep(1)
            if self.ser1.sameSignal == True:
                self.ms._statues_label.emit(statues_label)
                break
            num += 1
            if num >= target:
                self.ms._statues_label.emit("控制串口掉线")
                self.ms._print.emit(glo_var.sample_time + num)
                print("控制串口掉线")
                self.ms._show_error_message.emit("控制串口掉线, 请重新操作")
                break
