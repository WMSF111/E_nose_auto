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
        glo_var.app_state.update_current_temp(0)
        self._running = False  # 设置读取标志为 False
        self.time = 0
        self._stop_evt.set()  # 通过事件通知线程停止
        if name != None:
            print(name + "线程已结束")
        else:
            print("线程已结束")

class Serial1opea():
    # 一系列串口控制硬件的操作
    def __init__(self, ms, ser, ser1 = None):
        self.ser = ser
        self.ser1 = ser1
        self.ms = ms
        # self.time_th = None
        self._running =  True

    # 判断当前是否处于演示模式。
    def _is_demo_mode(self):
        return glo_var.app_state.demo_mode

    # 获取演示模式下单步等待时间。
    def _demo_seconds(self):
        return glo_var.app_state.demo_step_seconds

    # 根据流程类型获取演示模式下的等待时间。
    def _demo_duration(self, phase):
        phase_duration_map = {
            "base_clear": glo_var.app_state.base_time_value,
            "sample_collect": glo_var.app_state.sample_time_value,
            "room_clear": glo_var.app_state.exhaust_time_value,
            "heat_up": 0,
            "control": 0,
        }
        configured_seconds = phase_duration_map.get(phase, 0)
        return max(1, int(configured_seconds or self._demo_seconds()))

    # 在演示模式下等待固定时间后继续。
    def _demo_wait(self, status_text, phase="control", value=None, draw_open=False, draw_close=False):
        self.ms._statues_label.emit(status_text)
        if draw_open:
            self.ms._draw_open.emit()
        if draw_close:
            self.ms._draw_close.emit()
        wait_seconds = self._demo_duration(phase)
        for _ in range(wait_seconds):
            if self._running == False:
                return False
            time.sleep(1)
        if value is not None:
            self.ms._print.emit(value)
        return True

    # 向主串口发送指令。
    def _write_main(self, text):
        if self.ser and hasattr(self.ser, "write"):
            self.ser.write(text)

    # 向副串口发送指令。
    def _send_aux(self, *args, **kwargs):
        if self.ser1 and hasattr(self.ser1, "serialSend"):
            self.ser1.serialSend(*args, **kwargs)

    def GetSigal1(self, text): # ser1获取信号
        parts = text.split()  # ['55','AA', ...]
        Frame = frame_data.FrameData()
        parts = parts[2 : Frame.pkgLen - 1]
        Frame.setDataToArray(parts)
        if (Frame.buf[2] == '02'): # 获取温度
            num = int.from_bytes(bytes.fromhex(Frame.buf[4] + Frame.buf[5]), byteorder='big')  # 1000
            text = num / 10.0
            glo_var.app_state.update_current_temp(text) # 更新现在的温度

    def loop_to_target_temp(self, the_time = 0): # 每1s读取一次温度，直到达到合适温度glo_var.target_temp
        if self._is_demo_mode():
            target_temp = glo_var.app_state.target_temp or 25
            wait_seconds = self._demo_duration("heat_up")
            for second in range(1, wait_seconds + 1):
                if self._running == False:
                    return
                current_temp = target_temp * second / wait_seconds
                glo_var.app_state.update_current_temp(current_temp)
                self.ms._Currtem_spinBox.emit(glo_var.app_state.current_temp)
                time.sleep(1)
            self.ms._attendtime_spinBox.emit(wait_seconds)
            return
        while True:  # 没到回复
            if self._running == False:
                break
            # 线程安全：把值读出来再比较
            print("现在温度是：", glo_var.app_state.current_temp)
            self.ms._Currtem_spinBox.emit(glo_var.app_state.current_temp)
            self.ser1.serialSend(2, glo_var.app_state.channels[1])
            the_time += 1
            if glo_var.app_state.target_temp <= glo_var.app_state.current_temp:  # 当目标温度达成
                print("达到目标温度需要时间：",the_time)
                self.ms._attendtime_spinBox.emit(the_time)
                break
            time.sleep(1)

    def autosample(self):
        self.base_clear()
        self.loop_to_target_temp()# 循环到达指定温度
        self.now_Sam = 1
        while self.now_Sam <= glo_var.app_state.target_sample_count and self._running == True:  # target_Sam个样品循环操作
            glo_var.app_state.current_sample_index = self.now_Sam
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
        if self._is_demo_mode():
            wait_seconds = self._demo_duration("base_clear")
            if self._demo_wait("开始进行基线处理", phase="base_clear"):
                self.ms._statues_label.emit("基线处理完成")
                self.ms._print.emit(wait_seconds)
            self.ms._Clear_Button.emit(True)
            return
        text = ("11\n\r")
        num = 0
        while True:  # 没到回复
            if self._running == False:
                break
            self._write_main(text)  # 开始采集信号
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
            if num >= glo_var.app_state.base_time_value + 5:
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
        if self._is_demo_mode():
            if self._demo_wait("样品开始采集", phase="sample_collect", draw_open=True):
                self.ms._statues_label.emit("采样处理完成")
                self.ms._draw_close.emit()
            self.ms._Collectbegin_Button.emit(True)
            return
        if glo_var.app_state.auto_mode == True:
            self.ser_opea(3, glo_var.app_state.sample_time_value, re="55 AA 03 00 01 00 00 00 00 0A") # 信号采集
        while True:  # 没到回复
            if self._running == False:
                break
            self._write_main(text)  # 开始采集信号
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
            if num >= glo_var.app_state.sample_time_value + 5:
                self.ms._statues_label.emit("采样时长超时")
                self.ms._draw_close.emit()
                self.ms._show_error_message.emit(f"采样时长超时{glo_var.app_state.sample_time_value + 5}, 请重新操作")
                self.stop()
                break
        self.ms._Collectbegin_Button.emit(True)


    def room_clear(self):
        if self._running == True:
            self.ms._Clearroom_Button.emit(False)
        text = ("31\n\r")
        num = 0
        if self._is_demo_mode():
            room_clear_seconds = self._demo_duration("room_clear")
            if self._demo_wait(
                "气室正在清洗",
                phase="room_clear",
                value=glo_var.app_state.sample_time_value + room_clear_seconds,
            ):
                self.ms._statues_label.emit("洗气处理完成")
            self.ms._Clearroom_Button.emit(True)
            return
        if glo_var.app_state.auto_mode == True:
            self.ser_opea(4, glo_var.app_state.exhaust_time_value)
        while True:  # 没到回复
            if self._running == False:
                break
            self._write_main(text)  # 开始采集信号
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
                self.ms._print.emit(glo_var.app_state.sample_time_value + num)
                break
            if num >= glo_var.app_state.exhaust_time_value + 5:
                self.ms._statues_label.emit("洗气时长超时")
                self.ms._Clearroom_Button.emit(True)
                self.ms._show_error_message.emit(str("洗气时长超时：", num, ", 请重新操作"))
                self.stop()
                break
        self.ms._Clearroom_Button.emit(True)

    def stop(self, name = "Serial1opea"):
        self._running = False  # 设置读取标志为 False
        # 确保串口恢复正常状态
        if hasattr(self.ser, 'resume'):
            self.ser.resume()
        if hasattr(self.ser1, 'resume'):
            self.ser1.resume()
        if name != None:
            print(name + "线程已结束")
        else:
            print("线程已结束")

    def pos_top(self):
        if(self.now_Sam != glo_var.app_state.target_sample_count):
            print("转移位置到下一个样品" + str(self.now_Sam + 1))
            self.ser_opea("0A", glo_var.app_state.positions[self.now_Sam + 1][0], glo_var.app_state.positions[self.now_Sam + 1][1],
                                (int)(glo_var.app_state.positions[self.now_Sam + 1][2] * 0.1), target = 10)
        else:
            self.Stra()
            self.ms._Collectbegin_Button.emit(True)

    def pos_top_before(self):
        print("拔出样品" + str(self.now_Sam))
        self.ser_opea("0A", glo_var.app_state.positions[self.now_Sam][0], glo_var.app_state.positions[self.now_Sam][1],
                                (int)(glo_var.app_state.positions[self.now_Sam][2] * 0.1), target = 10)

    def pos_down(self):
        self.ser_opea("0A", glo_var.app_state.positions[self.now_Sam][0], glo_var.app_state.positions[self.now_Sam][1],
                                (int)(glo_var.app_state.positions[self.now_Sam][2]), target = 20)
        print("插入样品" + str(self.now_Sam))

    def Stra(self): # 运动轴回到原点
        self.ser_opea('0C')

    def ser_opea(self, opea, opea1 = 0, opea2 = 0, opea3 = 0, re = None, target = 5, statues_label = "处理完成"): # ser与ser1发送信号
        if self._is_demo_mode():
            self._demo_wait(statues_label, phase="control")
            return
        num = 0
        while True:  # ser1串口输入
            if self._running == False:
                break
            self._send_aux(opea, opea1, opea2, opea3, re = re)
            time.sleep(1)
            if self.ser1.sameSignal == True:
                self.ms._statues_label.emit(statues_label)
                break
            num += 1
            if num >= target:
                self.ms._statues_label.emit("控制串口掉线")
                self.ms._print.emit(glo_var.app_state.sample_time_value + num)
                print("控制串口掉线")
                self.ms._show_error_message.emit("控制串口掉线, 请重新操作")
                break
