import os
import pandas as pd
import numpy as np

import re
from datetime import datetime

from PySide6.QtCore import Signal, QObject


class PreprocessWorker(QObject):
    """预处理工作线程"""
    progress = Signal(int, str)  # 进度百分比, 状态信息
    finished = Signal(bool, str)  # 成功标志, 消息
    file_processed = Signal(str, str)  # 文件名, 状态

    def __init__(self, files, data_dir, file_type, target_col, data_columns,
                 filter_func, value_func, parent=None):
        super().__init__(parent)
        self.files = files
        self.data_dir = data_dir
        self.file_type = file_type
        self.target_col = target_col
        self.data_columns = data_columns
        self.filter_func = filter_func
        self.value_func = value_func
        self.running = True

    def run(self):
        """执行预处理任务"""
        try:
            total_files = len(self.files)
            if total_files == 0:
                self.finished.emit(False, "没有选择任何文件")
                return

            if self.file_type == "CSV":
                if self.target_col == "--请选择--" or not self.target_col:
                    self.finished.emit(False, "CSV文件类型必须选择target列")
                    return

                if not self.data_columns or len(self.data_columns) == 0:
                    self.finished.emit(False, "CSV文件类型必须选择至少一个数据列")
                    return

                if self.target_col in self.data_columns:
                    self.finished.emit(False, f"数据列中不能包含target列 '{self.target_col}'")
                    return

            processed_count = 0
            error_files = []

            for idx, file_path in enumerate(self.files):
                if not self.running:
                    break

                try:
                    file_name = os.path.basename(file_path)
                    self.progress.emit(int((idx + 1) * 100 / total_files),
                                       f"正在处理: {file_name}")

                    if self.file_type == "TXT":
                        success, new_file = self._process_txt_file(file_path)
                    else:  # CSV
                        success, new_file = self._process_csv_file(file_path)

                    if success:
                        processed_count += 1
                        self.file_processed.emit(file_name, "成功")
                    else:
                        error_files.append(file_name)
                        self.file_processed.emit(file_name, "失败")

                except Exception as e:
                    error_files.append(os.path.basename(file_path))
                    self.file_processed.emit(os.path.basename(file_path), f"错误: {str(e)}")

            if self.running:
                if processed_count == total_files:
                    msg = f"所有文件处理完成，共{processed_count}个文件"
                    self.finished.emit(True, msg)
                else:
                    msg = f"处理完成，成功{processed_count}个，失败{len(error_files)}个"
                    if error_files:
                        msg += f"\n失败文件: {', '.join(error_files[:5])}"
                        if len(error_files) > 5:
                            msg += f"...等{len(error_files)}个文件"
                    self.finished.emit(False, msg)

        except Exception as e:
            self.finished.emit(False, f"处理过程中发生错误: {str(e)}")

    def stop(self):
        self.running = False

    def _process_txt_file(self, file_path):
        """处理TXT文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            data = []
            for line in lines:
                line = line.strip()
                if line:  # 跳过空行
                    parts = re.split(r'\s+', line)
                    row_data = []
                    for part in parts:
                        try:
                            value = float(part)
                            row_data.append(value)
                        except ValueError:
                            raise ValueError(f"发现非数值数据: {part}")
                    data.append(row_data)

            if not data:
                raise ValueError("文件为空或格式不正确")

            data_lengths = [len(row) for row in data]
            if len(set(data_lengths)) > 1:
                raise ValueError(f"数据行长度不一致: {data_lengths}")

            data_array = np.array(data)
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            if self.filter_func == "--请选择--" and self.value_func == "--请选择--":
                raise ValueError("请至少选择一种处理方法（滤波函数或值选择）")

            # 预处理数据
            # TXT文件：使用文件名作为target
            processed_data = self._apply_preprocessing(data_array, has_target=True, target=file_name)

            # 生成新文件名 - TXT格式
            new_filename = self._generate_new_filename(file_path, "txt")
            new_filepath = os.path.join(self.data_dir, new_filename)

            # 保存处理后的数据 - 纯数据格式
            self._save_txt_file(processed_data, new_filepath)

            return True, new_filepath

        except Exception as e:
            raise Exception(f"处理TXT文件失败: {str(e)}")

    def _process_csv_file(self, file_path):
        """处理CSV文件"""
        try:
            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError("CSV文件为空")

            if self.target_col not in df.columns:
                raise ValueError(f"目标列 '{self.target_col}' 不在CSV文件中")

            for col in self.data_columns:
                if col not in df.columns:
                    raise ValueError(f"数据列 '{col}' 不在CSV文件中")

                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if df[col].isna().any():
                            raise ValueError(f"数据列 '{col}' 包含非数值数据")
                    except:
                        raise ValueError(f"数据列 '{col}' 包含非数值数据，无法转换为数值类型")

            if self.filter_func == "--请选择--" and self.value_func == "--请选择--":
                raise ValueError("请至少选择一种处理方法（滤波函数或值选择）")

            feature_cols = [col for col in self.data_columns if col != self.target_col]
            if not feature_cols:
                raise ValueError("请至少选择一个非target的数据列")

            features = df[feature_cols].values
            target = df[self.target_col].values

            processed_features, processed_target = self._apply_preprocessing(features, has_target=True, target=target,
                                                                             is_csv=True)

            if self.value_func != "--请选择--":
                if processed_features.ndim == 1:
                    processed_features = processed_features.reshape(-1, 1)

                processed_df = pd.DataFrame(processed_features, columns=feature_cols)
                processed_df[self.target_col] = processed_target
            else:
                # 只有滤波处理：保持原始行数
                processed_df = pd.DataFrame(processed_features, columns=feature_cols)
                processed_df[self.target_col] = target

            # 生成CSV文件
            new_filename = self._generate_new_filename(file_path, "csv")
            new_filepath = os.path.join(self.data_dir, new_filename)
            processed_df.to_csv(new_filepath, index=False, encoding='utf-8')

            return True, new_filepath

        except Exception as e:
            raise Exception(f"处理CSV文件失败: {str(e)}")

    def _apply_preprocessing(self, data, has_target=False, target=None, is_csv=False):
        """应用预处理函数
        Args:
            data: 要处理的数据
            has_target: 是否有target列
            target: target数据
            is_csv: 是否是CSV文件处理
        Returns:
            处理后的数据和target（对于CSV值选择处理）
        """
        processed_data = data.copy()
        processed_target = target

        # 滤波函数处理
        if self.filter_func != "--请选择--":
            if self.filter_func == "算术平均滤波法":
                processed_data = self._arithmetic_mean_filter(processed_data)
            elif self.filter_func == "递推平均滤波法":
                processed_data = self._recursive_mean_filter(processed_data)
            elif self.filter_func == "中位值平均滤波法":
                processed_data = self._median_average_filter(processed_data)
            elif self.filter_func == "一阶滞后滤波法":
                processed_data = self._first_order_lag_filter(processed_data)
            elif self.filter_func == "加权递推平均滤波法":
                processed_data = self._weighted_recursive_average_filter(processed_data)
            elif self.filter_func == "消抖滤波法":
                processed_data = self._shake_off_filter(processed_data)
            elif self.filter_func == "限幅消抖滤波法":
                processed_data = self._amplitude_limiting_shake_off_filter(processed_data)

        # 值选择处理
        if self.value_func != "--请选择--":
            if not has_target:
                # 对于没有target的数据，值选择无法应用
                raise ValueError("值选择需要target列，但当前数据没有target列")

            # 检查target类型
            if isinstance(target, str):
                # TXT文件：target是文件名，所有数据都属于同一个target
                # 对于TXT文件，整个文件作为一个组
                if self.value_func == "平均值":
                    processed_data = self._calculate_mean(processed_data)
                elif self.value_func == "中位值":
                    processed_data = self._calculate_median(processed_data)
                elif self.value_func == "众数":
                    processed_data = self._calculate_mode(processed_data)
                elif self.value_func == "极差":
                    processed_data = self._calculate_range(processed_data)
                elif self.value_func == "最大值":
                    processed_data = self._calculate_max(processed_data)
                elif self.value_func == "最大斜率":
                    processed_data = self._calculate_max_slope(processed_data)
            else:
                # CSV文件：target是数组，需要分组
                if is_csv:
                    # CSV文件处理
                    if self.value_func == "平均值":
                        processed_data, processed_target = self._calculate_group_mean(processed_data, target)
                    elif self.value_func == "中位值":
                        processed_data, processed_target = self._calculate_group_median(processed_data, target)
                    elif self.value_func == "众数":
                        processed_data, processed_target = self._calculate_group_mode(processed_data, target)
                    elif self.value_func == "极差":
                        processed_data, processed_target = self._calculate_group_range(processed_data, target)
                    elif self.value_func == "最大值":
                        processed_data, processed_target = self._calculate_group_max(processed_data, target)
                    elif self.value_func == "最大斜率":
                        processed_data, processed_target = self._calculate_group_max_slope(processed_data, target)
                else:
                    # 其他情况
                    if self.value_func == "平均值":
                        processed_data = self._calculate_group_mean_simple(processed_data, target)
                    elif self.value_func == "中位值":
                        processed_data = self._calculate_group_median_simple(processed_data, target)
                    elif self.value_func == "众数":
                        processed_data = self._calculate_group_mode_simple(processed_data, target)
                    elif self.value_func == "极差":
                        processed_data = self._calculate_group_range_simple(processed_data, target)
                    elif self.value_func == "最大值":
                        processed_data = self._calculate_group_max_simple(processed_data, target)
                    elif self.value_func == "最大斜率":
                        processed_data = self._calculate_group_max_slope_simple(processed_data, target)

        if is_csv and self.value_func != "--请选择--":
            return processed_data, processed_target
        else:
            return processed_data

    # 滤波函数实现
    def _arithmetic_mean_filter(self, data, window_size=2):
        """算术平均滤波法"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            column = data[:, i].copy()
            filtered = np.zeros_like(column)
            for j in range(len(column)):
                start = max(0, j - window_size + 1)
                end = j + 1
                filtered[j] = np.mean(column[start:end])
            result[:, i] = filtered
        return result

    def _recursive_mean_filter(self, data, window_size=2):
        """递推平均滤波法"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            column = data[:, i].copy()
            filtered = np.zeros_like(column)
            for j in range(len(column)):
                if j == 0:
                    filtered[j] = column[j]
                else:
                    start = max(0, j - window_size + 1)
                    end = j + 1
                    filtered[j] = np.mean(column[start:end])
            result[:, i] = filtered
        return result

    def _median_average_filter(self, data, window_size=2):
        """中位值平均滤波法"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            column = data[:, i].copy()
            filtered = np.zeros_like(column)
            for j in range(len(column)):
                start = max(0, j - window_size + 1)
                end = j + 1
                window_data = column[start:end]
                # 先取中位值，再取平均
                median_val = np.median(window_data)
                filtered[j] = np.mean([median_val, column[j]])
            result[:, i] = filtered
        return result

    def _first_order_lag_filter(self, data, alpha=0.9):
        """一阶滞后滤波法"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            column = data[:, i].copy()
            filtered = np.zeros_like(column)
            filtered[0] = column[0]
            for j in range(1, len(column)):
                filtered[j] = alpha * filtered[j - 1] + (1 - alpha) * column[j]
            result[:, i] = filtered
        return result

    def _weighted_recursive_average_filter(self, data, alpha=0.9):
        """加权递推平均滤波法"""
        return self._first_order_lag_filter(data, alpha)  # 与一阶滞后类似

    def _shake_off_filter(self, data, threshold=4):
        """消抖滤波法"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            column = data[:, i].copy()
            filtered = np.zeros_like(column)
            last_value = column[0]
            counter = 0

            for j in range(len(column)):
                if abs(column[j] - last_value) < threshold:
                    filtered[j] = column[j]
                    last_value = column[j]
                    counter = 0
                else:
                    counter += 1
                    if counter >= threshold:
                        filtered[j] = column[j]
                        last_value = column[j]
                        counter = 0
                    else:
                        filtered[j] = last_value
            result[:, i] = filtered
        return result

    def _amplitude_limiting_shake_off_filter(self, data, amplitude_limit=200, shake_threshold=3):
        """限幅消抖滤波法"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            column = data[:, i].copy()
            filtered = np.zeros_like(column)
            last_value = column[0]
            counter = 0

            for j in range(len(column)):
                # 限幅处理
                if abs(column[j] - last_value) > amplitude_limit:
                    column[j] = last_value

                # 消抖处理
                if abs(column[j] - last_value) < shake_threshold:
                    filtered[j] = column[j]
                    last_value = column[j]
                    counter = 0
                else:
                    counter += 1
                    if counter >= shake_threshold:
                        filtered[j] = column[j]
                        last_value = column[j]
                        counter = 0
                    else:
                        filtered[j] = last_value
            result[:, i] = filtered
        return result

    # 值选择函数实现 - 用于TXT文件（整个文件作为一个组）
    def _calculate_mean(self, data):
        """计算整个数据的平均值"""
        if len(data.shape) == 1:
            return np.array([[np.mean(data)]])
        else:
            return np.array([np.mean(data, axis=0)])

    def _calculate_median(self, data):
        """计算整个数据的中位值"""
        if len(data.shape) == 1:
            return np.array([[np.median(data)]])
        else:
            return np.array([np.median(data, axis=0)])

    def _calculate_mode(self, data):
        """计算整个数据的众数"""
        if len(data.shape) == 1:
            # 对于一维数组
            values, counts = np.unique(data, return_counts=True)
            mode_value = values[np.argmax(counts)]
            return np.array([[mode_value]])
        else:
            # 对于多维数组，对每列计算众数
            modes = []
            for i in range(data.shape[1]):
                column = data[:, i]
                values, counts = np.unique(column, return_counts=True)
                mode_value = values[np.argmax(counts)]
                modes.append(mode_value)
            return np.array([modes])

    def _calculate_range(self, data):
        """计算整个数据的极差"""
        if len(data.shape) == 1:
            return np.array([[np.max(data) - np.min(data)]])
        else:
            return np.array([np.max(data, axis=0) - np.min(data, axis=0)])

    def _calculate_max(self, data):
        """计算整个数据的最大值"""
        if len(data.shape) == 1:
            return np.array([[np.max(data)]])
        else:
            return np.array([np.max(data, axis=0)])

    def _calculate_max_slope(self, data):
        """计算整个数据的最大斜率"""
        if len(data.shape) == 1:
            if len(data) <= 1:
                return np.array([[0]])
            slopes = np.abs(np.diff(data))
            max_slope = np.max(slopes) if len(slopes) > 0 else 0
            return np.array([[max_slope]])
        else:
            max_slopes = []
            for i in range(data.shape[1]):
                column = data[:, i]
                if len(column) <= 1:
                    max_slopes.append(0)
                else:
                    slopes = np.abs(np.diff(column))
                    max_slope = np.max(slopes) if len(slopes) > 0 else 0
                    max_slopes.append(max_slope)
            return np.array([max_slopes])

    # 值选择函数实现 - 用于CSV文件（需要分组）返回处理后的数据和target
    def _calculate_group_mean(self, data, target):
        """按target分组计算平均值"""
        df = pd.DataFrame(data)
        df['target'] = target
        grouped = df.groupby('target').mean()
        return grouped.values, grouped.index.values

    def _calculate_group_median(self, data, target):
        """按target分组计算中位值"""
        df = pd.DataFrame(data)
        df['target'] = target
        grouped = df.groupby('target').median()
        return grouped.values, grouped.index.values

    def _calculate_group_mode(self, data, target):
        """按target分组计算众数"""
        df = pd.DataFrame(data)
        df['target'] = target

        def mode_func(x):
            modes = x.mode()
            return modes.iloc[0] if len(modes) > 0 else x.iloc[0]

        grouped = df.groupby('target').agg(mode_func)
        return grouped.values, grouped.index.values

    def _calculate_group_range(self, data, target):
        """按target分组计算极差"""
        df = pd.DataFrame(data)
        df['target'] = target

        def range_func(x):
            return x.max() - x.min()

        grouped = df.groupby('target').agg(range_func)
        return grouped.values, grouped.index.values

    def _calculate_group_max(self, data, target):
        """按target分组计算最大值"""
        df = pd.DataFrame(data)
        df['target'] = target
        grouped = df.groupby('target').max()
        return grouped.values, grouped.index.values

    def _calculate_group_max_slope(self, data, target):
        """按target分组计算最大斜率"""
        df = pd.DataFrame(data)
        df['target'] = target

        def max_slope_func(x):
            if len(x) <= 1:
                return 0
            # 计算相邻数据点的斜率
            slopes = np.abs(np.diff(x))
            return np.max(slopes) if len(slopes) > 0 else 0

        grouped = df.groupby('target').agg(max_slope_func)
        return grouped.values, grouped.index.values

    # 值选择函数实现 - 简单版本（不返回target）
    def _calculate_group_mean_simple(self, data, target):
        """按target分组计算平均值（简单版本）"""
        df = pd.DataFrame(data)
        df['target'] = target
        grouped = df.groupby('target').mean()
        return grouped.values

    def _calculate_group_median_simple(self, data, target):
        """按target分组计算中位值（简单版本）"""
        df = pd.DataFrame(data)
        df['target'] = target
        grouped = df.groupby('target').median()
        return grouped.values

    def _calculate_group_mode_simple(self, data, target):
        """按target分组计算众数（简单版本）"""
        df = pd.DataFrame(data)
        df['target'] = target

        def mode_func(x):
            modes = x.mode()
            return modes.iloc[0] if len(modes) > 0 else x.iloc[0]

        grouped = df.groupby('target').agg(mode_func)
        return grouped.values

    def _calculate_group_range_simple(self, data, target):
        """按target分组计算极差（简单版本）"""
        df = pd.DataFrame(data)
        df['target'] = target

        def range_func(x):
            return x.max() - x.min()

        grouped = df.groupby('target').agg(range_func)
        return grouped.values

    def _calculate_group_max_simple(self, data, target):
        """按target分组计算最大值（简单版本）"""
        df = pd.DataFrame(data)
        df['target'] = target
        grouped = df.groupby('target').max()
        return grouped.values

    def _calculate_group_max_slope_simple(self, data, target):
        """按target分组计算最大斜率（简单版本）"""
        df = pd.DataFrame(data)
        df['target'] = target

        def max_slope_func(x):
            if len(x) <= 1:
                return 0
            # 计算相邻数据点的斜率
            slopes = np.abs(np.diff(x))
            return np.max(slopes) if len(slopes) > 0 else 0

        grouped = df.groupby('target').agg(max_slope_func)
        return grouped.values

    def _generate_new_filename(self, original_path, file_type):
        """生成新的文件名"""
        original_name = os.path.splitext(os.path.basename(original_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 添加处理类型信息
        process_types = []
        if self.filter_func != "--请选择--":
            # 简化滤波函数名称
            filter_name_map = {
                "算术平均滤波法": "算术平均",
                "递推平均滤波法": "递推平均",
                "中位值平均滤波法": "中位值平均",
                "一阶滞后滤波法": "一阶滞后",
                "加权递推平均滤波法": "加权递推",
                "消抖滤波法": "消抖",
                "限幅消抖滤波法": "限幅消抖"
            }
            short_name = filter_name_map.get(self.filter_func, self.filter_func[:4])
            process_types.append(short_name)
        if self.value_func != "--请选择--":
            # 简化值选择名称
            value_name_map = {
                "平均值": "均值",
                "中位值": "中位",
                "众数": "众数",
                "极差": "极差",
                "最大值": "最大",
                "最大斜率": "最大斜率"
            }
            short_name = value_name_map.get(self.value_func, self.value_func[:2])
            process_types.append(short_name)

        process_str = "_".join(process_types) if process_types else "processed"

        new_name = f"{original_name}_{process_str}_{timestamp}.{file_type}"
        return new_name

    def _save_txt_file(self, data, filepath):
        """保存TXT文件 - 纯数据格式，每行用空格分隔"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for row in data:
                # 将每行数据用空格连接，保留适当精度
                if len(data.shape) == 1:
                    # 一维数据
                    line = f"{row:.6f}" if isinstance(row, (float, np.floating)) else str(row)
                    f.write(line + '\n')
                else:
                    # 多维数据
                    line = ' '.join([f"{x:.6f}" if isinstance(x, (float, np.floating)) else str(x) for x in row])
                    f.write(line + '\n')