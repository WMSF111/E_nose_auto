import os
import pandas as pd
import numpy as np

import re
from datetime import datetime
import traceback
import data_file.model_LDA
import data_file.model_SVM
import data_file.model_RF
import data_file.model_ANN
import data_file.model_KNN
import data_file.model_PLSR
import data_file.model_MLR
import data_file.model_SVR
import data_file.model_PCA
import data_file.model_LDA2
import data_file.model_LLE


from PySide6.QtCore import Signal, QObject, QThread

class ModelWorker(QThread):
    """线程运行算法"""
    # 定义信号
    finished = Signal(str)      # 成功完成，传递结果
    error = Signal(str)         # 发生错误，传递错误信息

    def __init__(self, dir_path, source_file_path, group_name, model_name, params, dpi):
        super().__init__()
        self.dir_path = dir_path            # 工作目录
        self.src_path = source_file_path    # 原文件路径
        self.group_name = group_name        # 模型分类
        self.model_name = model_name        # 模型名称
        self.params = params                # 参数
        self._is_running = True             # 线程是否运行
        self.dpi = dpi                      # 显示dpi

    def run(self):
        """线程执行"""
        try:
            df = pd.read_csv(self.src_path)         # 读取数据文件
            result = {}                             # 算法结果

            if self.group_name == "分类模型":
                if self.model_name == "LDA":
                    result = data_file.model_LDA.run(df, self.dir_path, self.params, self.dpi)
                elif self.model_name == "SVM":
                    result = data_file.model_SVM.run(df, self.dir_path, self.params, self.dpi)
                elif self.model_name == "RF":
                    result = data_file.model_RF.run(df, self.dir_path, self.params, self.dpi)
            #
            elif self.group_name == "预测模型":
                if self.model_name == "KNN":
                    result = data_file.model_KNN.run(df, self.dir_path, self.params, self.dpi)
                elif self.model_name == "BP-ANN":
                    result = data_file.model_ANN.run(df, self.dir_path, self.params, self.dpi)
                elif self.model_name == "PLSR":
                    result = data_file.model_PLSR.run(df, self.dir_path, self.params, self.dpi)
                elif self.model_name == "MLR":
                    result = data_file.model_MLR.run(df, self.dir_path, self.params, self.dpi)
                elif self.model_name == "SVR":
                    result = data_file.model_SVR.run(df, self.dir_path, self.params, self.dpi)

            elif self.group_name == "降维模型":
                if self.model_name == "PCA":
                    result = data_file.model_PCA.run(df, self.dir_path, self.params, self.dpi)
                elif self.model_name == "LDA":
                    result = data_file.model_LDA2.run(df, self.dir_path, self.params, self.dpi)
                elif self.model_name == "LLE":
                    result = data_file.model_LLE.run(df, self.dir_path, self.params, self.dpi)

            self.finished.emit(result)  # 任务完成，发送结果
        except Exception as e:
            # 捕获异常并发送错误信息
            error_msg = f"算法执行出错:\n\n {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def stop(self):
        """停止线程"""
        self._is_running = False
        self.quit()
        self.wait()