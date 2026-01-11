# -*- coding: utf-8 -*-

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QTreeWidgetItem, QMessageBox, QFileDialog, QMenu, QApplication, QWidget, QVBoxLayout, \
    QTextEdit, QPushButton, QLabel, QListWidget, QHBoxLayout, QListWidgetItem, QTableWidget, QTableWidgetItem, \
    QInputDialog, QLineEdit
import os
import tempfile
import pandas as pd


def load_txt_file_to_table(file_path, table_widget):
    """
    加载TXT文件到表格，每行按空格分隔
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

    max_columns = 0
    processed_lines = []

    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        if line.strip():
            fields = line.split()
            processed_lines.append(fields)
            max_columns = max(max_columns, len(fields))
        else:
            processed_lines.append(['[空行]'])
            max_columns = max(max_columns, 1)

    if not processed_lines:
        table_widget.setRowCount(0)
        table_widget.setColumnCount(0)
        return

    table_widget.setRowCount(len(processed_lines))
    table_widget.setColumnCount(max_columns)

    # 设置列标题（使用字母A, B, C...）
    column_headers = []
    for i in range(max_columns):
        column_headers.append(chr(65 + i) if i < 26 else f"Col{i + 1}")
    table_widget.setHorizontalHeaderLabels(column_headers)

    row_headers = []
    for i in range(len(processed_lines)):
        row_headers.append(str(i + 1))
    table_widget.setVerticalHeaderLabels(row_headers)

    for row_idx, fields in enumerate(processed_lines):
        for col_idx in range(max_columns):
            if col_idx < len(fields):
                cell_value = fields[col_idx]
            else:
                cell_value = ""

            item = QTableWidgetItem(cell_value)

            if cell_value == '[空行]':
                item.setForeground(QColor("#888888"))
                item.setBackground(QColor("#f8f8f8"))

            try:
                float(cell_value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            except ValueError:
                pass

            table_widget.setItem(row_idx, col_idx, item)


def load_csv_file_to_table(file_path, table_widget):
    """
    加载CSV文件到表格
    """
    try:
        df = pd.read_csv(file_path)
        row_count, col_count = df.shape

        table_widget.setRowCount(row_count)
        table_widget.setColumnCount(col_count)
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())
        row_headers = [str(i + 1) for i in range(row_count)]
        table_widget.setVerticalHeaderLabels(row_headers)
        for i in range(row_count):
            for j in range(col_count):
                cell_value = str(df.iloc[i, j])
                item = QTableWidgetItem(cell_value)
                try:
                    float_val = float(cell_value)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    if float_val.is_integer():
                        item.setText(str(int(float_val)))
                except ValueError:
                    pass
                table_widget.setItem(i, j, item)
    except Exception as e:
        print(f"使用pandas读取失败: {e}，尝试普通方式读取")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    lines = f.readlines()
            except:
                with open(file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
        if not lines:
            table_widget.setRowCount(0)
            table_widget.setColumnCount(0)
            return

        max_columns = 0
        processed_lines = []

        for i, line in enumerate(lines):
            line = line.rstrip('\n')
            if line.strip():
                fields = [field.strip() for field in line.split(',')]
                processed_lines.append(fields)
                max_columns = max(max_columns, len(fields))
            else:
                processed_lines.append(['[空行]'])
                max_columns = max(max_columns, 1)

        if not processed_lines:
            table_widget.setRowCount(0)
            table_widget.setColumnCount(0)
            return
        table_widget.setRowCount(len(processed_lines))
        table_widget.setColumnCount(max_columns)
        if len(processed_lines) > 0:
            first_row = processed_lines[0]
            column_headers = []
            for i in range(max_columns):
                if i < len(first_row):
                    column_headers.append(first_row[i])
                else:
                    column_headers.append(chr(65 + i) if i < 26 else f"Col{i + 1}")
            data_rows = processed_lines[1:]
            table_widget.setHorizontalHeaderLabels(column_headers)
        else:
            data_rows = []
            column_headers = [chr(65 + i) if i < 26 else f"Col{i + 1}" for i in range(max_columns)]
            table_widget.setHorizontalHeaderLabels(column_headers)

        row_headers = [str(i + 1) for i in range(len(data_rows))]
        table_widget.setVerticalHeaderLabels(row_headers)
        table_widget.setRowCount(len(data_rows))
        for row_idx, fields in enumerate(data_rows):
            for col_idx in range(max_columns):
                if col_idx < len(fields):
                    cell_value = fields[col_idx]
                else:
                    cell_value = ""
                item = QTableWidgetItem(cell_value)

                if cell_value == '[空行]':
                    item.setForeground(QColor("#888888"))
                    item.setBackground(QColor("#f8f8f8"))
                try:
                    float(cell_value)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                except ValueError:
                    pass
                table_widget.setItem(row_idx, col_idx, item)


class LeftFrameManager:
    def __init__(self, ui_instance):
        """初始化左侧文件树管理器"""
        self.ui = ui_instance
        # 初始化变量
        self.imported_roots = {}  # 存储已导入的根节点
        self.checkbox_states = {}  # 存储复选框状态
        self.folder_checkbox_states = {}  # 存储文件夹复选框状态
        self.fixed_root_item = None  # 存储固定的根节点

        # 初始化固定根节点
        self.init_fixed_root()
        # 连接信号与槽
        self.setup_connections()

    def setup_connections(self):
        """设置信号与槽的连接"""
        # 连接按钮点击事件
        self.ui.btn_import.clicked.connect(self.import_folder)
        self.ui.treeWidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.treeWidget.customContextMenuRequested.connect(self.show_context_menu)
        self.ui.treeWidget.itemClicked.connect(self.on_tree_item_clicked)
        # 文件数上双击文件打开文件内容
        self.ui.treeWidget.itemDoubleClicked.connect(self.on_tree_itemDoubleClicked)

    def init_fixed_root(self):
        """初始化固定的根节点"""
        # 获取当前程序运行目录下的tmp目录
        current_dir = os.getcwd()
        tmp_dir = os.path.join(current_dir, "tmp")

        # 如果tmp目录不存在，创建它
        if not os.path.exists(tmp_dir):
            try:
                os.makedirs(tmp_dir, exist_ok=True)
            except Exception as e:
                print(f"创建tmp目录失败: {e}")
                tmp_dir = tempfile.gettempdir()  # 使用系统临时目录

        self.fixed_root_item = QTreeWidgetItem(self.ui.treeWidget)
        self.fixed_root_item.setText(0, "数据计算目录")
        self.fixed_root_item.setText(1, "□")  # 文件夹有复选框
        self.fixed_root_item.setData(0, Qt.ItemDataRole.UserRole, "fixed_root")  # 特殊标识符
        self.fixed_root_item.setData(0, Qt.ItemDataRole.UserRole + 1, tmp_dir)  # 存储实际路径
        self.fixed_root_item.setExpanded(True)

        # 设置tooltip显示完整路径
        self.fixed_root_item.setToolTip(0, f"路径: {tmp_dir}")
        self.fixed_root_item.setToolTip(1, f"路径: {tmp_dir}")
        self.folder_checkbox_states[self.fixed_root_item] = False

        # 添加路径作为子节点显示
        path_item = QTreeWidgetItem(self.fixed_root_item)
        path_item.setText(0, f"路径: {tmp_dir}")
        path_item.setText(1, "设置")  # 路径节点有选择功能
        path_item.setData(0, Qt.ItemDataRole.UserRole, "path_select")
        path_item.setData(0, Qt.ItemDataRole.UserRole + 1, tmp_dir)
        path_item.setToolTip(0, tmp_dir)
        path_item.setToolTip(1, "点击设置数据计算目录")
        path_item.setForeground(1, QColor("#0078d4"))

        self.scan_folder_files(tmp_dir, self.fixed_root_item)

    def import_folder(self):
        """导入文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            None,
            "选择要导入的文件夹",
            os.path.expanduser("~")
        )

        if folder_path:
            try:
                self.add_folder_to_tree(folder_path)
                self.update_stats()
                self.ui.bottom_manager.update_status(f"已导入: {os.path.basename(folder_path)}")
            except Exception as e:
                QMessageBox.critical(None, "错误", f"导入文件夹时出错:\n{str(e)}")

    def add_folder_to_tree(self, folder_path):
        """将文件夹添加到树形结构中"""
        folder_name = os.path.basename(folder_path)

        if folder_path in self.imported_roots:
            QMessageBox.information(None, "提示", f"文件夹 '{folder_name}' 已经导入过了")
            return

        # 创建根节点
        root_item = QTreeWidgetItem(self.ui.treeWidget)
        root_item.setText(0, f"{folder_name}")
        root_item.setText(1, "□")  # 文件夹有复选框
        root_item.setData(0, Qt.ItemDataRole.UserRole, "root")
        root_item.setData(0, Qt.ItemDataRole.UserRole + 1, folder_path)
        root_item.setExpanded(True)

        # 设置文件夹节点的tooltip显示完整路径
        root_item.setToolTip(0, folder_path)
        root_item.setToolTip(1, folder_path)
        self.folder_checkbox_states[root_item] = False

        # 存储根节点信息
        self.imported_roots[folder_path] = {
            'item': root_item,
            'path': folder_path,
            'name': folder_name
        }

        # 添加文件夹路径作为子节点显示
        path_item = QTreeWidgetItem(root_item)
        path_item.setText(0, f"路径: {folder_path}")
        path_item.setText(1, "")  # 路径节点没有复选框
        path_item.setData(0, Qt.ItemDataRole.UserRole, "path")
        path_item.setToolTip(0, folder_path)
        path_item.setToolTip(1, folder_path)

        self.scan_folder_files(folder_path, root_item)
        self.ui.treeWidget.expandItem(root_item)

    def scan_folder_files(self, folder_path, parent_item):
        """扫描文件夹下的文件"""
        try:
            items = list(os.scandir(folder_path))
            # 只添加文件，不添加子文件夹
            file_count = 0
            for item in items:
                if item.is_file() and not item.name.startswith('.'):  # 忽略隐藏文件
                    # 检查文件扩展名，只保留.txt和.csv文件
                    file_ext = os.path.splitext(item.name)[1].lower()
                    if file_ext in ['.txt', '.csv']:
                        self.add_file(item.path, parent_item)
                        file_count += 1
                    else:
                        print(f"跳过非TXT/CSV文件: {item.name}")
        except PermissionError:
            error_item = QTreeWidgetItem(parent_item)
            error_item.setText(0, "无访问权限")
            error_item.setText(1, "")  # 错误节点没有复选框
            error_item.setData(0, Qt.ItemDataRole.UserRole, "error")
        except Exception as e:
            error_item = QTreeWidgetItem(parent_item)
            error_item.setText(0, f"扫描错误: {str(e)}")
            error_item.setText(1, "")  # 错误节点没有复选框
            error_item.setData(0, Qt.ItemDataRole.UserRole, "error")

    def add_file(self, file_path, parent_item):
        """添加文件到树中"""
        file_name = os.path.basename(file_path)

        # 创建文件项
        file_item = QTreeWidgetItem(parent_item)
        file_item.setText(0, file_name)  # 只显示文件名，不加图标
        file_item.setText(1, "□")  # 初始复选框状态，使用更大更明显的符号
        file_item.setData(0, Qt.ItemDataRole.UserRole, "file")
        file_item.setData(0, Qt.ItemDataRole.UserRole + 1, file_path)  # 存储文件路径

        # 设置文件的tooltip显示完整路径
        file_item.setToolTip(0, file_path)
        file_item.setToolTip(1, file_path)
        self.checkbox_states[file_item] = False

    def on_tree_item_clicked(self, item, column):
        """处理树形项目的点击事件"""
        item_type = item.data(0, Qt.ItemDataRole.UserRole)

        # 处理第二列点击
        if column == 1:
            if item_type == "file":
                self.toggle_checkbox(item)
            elif item_type in ["root", "fixed_root"]:
                self.toggle_folder_checkbox(item)
            elif item_type == "path_select":
                self.select_data_directory(item)

    def on_tree_itemDoubleClicked(self, item, column):
        """
        双击文件树项目时的处理函数
        只处理txt和csv文件，显示在右侧标签页中
        """
        item_type = item.data(0, Qt.ItemDataRole.UserRole)
        file_path = item.data(0, Qt.ItemDataRole.UserRole + 1)
        file_ext = os.path.splitext(file_path)[1].lower()
        if item_type == "file":
            if file_ext not in ['.txt', '.csv']:
                self.ui.bottom_manager.update_status(f"不支持的文件类型: {file_ext}")
                return

            file_name = os.path.basename(file_path)
            tab_index = self.find_tab_by_file_path(file_path)
            if tab_index >= 0:
                # 如果文件已打开，切换到该标签页
                self.ui.tabWidget.setCurrentIndex(tab_index)
                self.ui.bottom_manager.update_status(f"已切换到: {file_name}")
                return
            try:
                new_tab = QWidget()
                new_tab.setObjectName(f"tab_{file_name}")
                layout = QVBoxLayout(new_tab)
                layout.setContentsMargins(5, 5, 5, 5)
                # label_title = QLabel(f"文件: {file_name}", new_tab)
                # label_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                # layout.addWidget(label_title)

                table_widget = QTableWidget(new_tab)
                table_widget.setObjectName(f"tableWidget_{file_name}")

                table_widget.setStyleSheet("""
                        QTableWidget {
                            font-family: Consolas, monospace;
                            font-size: 12px;
                            background-color: white;
                            border: 1px solid #ccc;
                            border-radius: 3px;
                            gridline-color: #e0e0e0;
                        }
                        QTableWidget::item {
                            padding: 2px;
                        }
                        QTableWidget::item:selected {
                            background-color: #e0f0ff;
                            color: #333;
                        }
                        QHeaderView::section {
                            background-color: #f0f0f0;
                            padding: 5px;
                            border: 1px solid #ccc;
                            font-weight: bold;
                        }
                    """)

                table_widget.setAlternatingRowColors(True)
                table_widget.setSortingEnabled(True)
                table_widget.horizontalHeader().setStretchLastSection(True)

                # 读取文件内容并填充表格
                if file_ext == '.txt':
                    load_txt_file_to_table(file_path, table_widget)
                elif file_ext == '.csv':
                    load_csv_file_to_table(file_path, table_widget)

                layout.addWidget(table_widget)

                info_layout = QHBoxLayout()
                row_count = table_widget.rowCount()
                col_count = table_widget.columnCount()
                count_label = QLabel(f"行数: {row_count}, 列数: {col_count}", new_tab)
                count_label.setFont(QFont("Arial", 10))
                info_layout.addWidget(count_label)

                path_label = QLabel(f"路径: {file_path}", new_tab)
                path_label.setFont(QFont("Arial", 9))
                path_label.setStyleSheet("color: #666666;")
                info_layout.addWidget(path_label)

                info_layout.addStretch()

                btn_close = QPushButton(f"关闭", new_tab)
                btn_close.setFixedSize(80, 30)
                btn_close.clicked.connect(lambda: self.close_tab(new_tab))
                info_layout.addWidget(btn_close)

                layout.addLayout(info_layout)

                new_tab.file_path = file_path
                new_tab.table_widget = table_widget

                tab_index = self.ui.tabWidget.addTab(new_tab, file_name)
                self.ui.tabWidget.setCurrentIndex(tab_index)

                self.ui.statusBar.setText(f"已打开: {file_name} ({row_count}行×{col_count}列)")

            except Exception as e:
                # 文件读取失败
                self.ui.statusBar.setText(f"打开文件失败: {str(e)}")
                import traceback
                print(traceback.format_exc())

    def find_tab_by_file_path(self, file_path):
        """
        根据文件路径查找是否已打开对应的标签页
        """
        for i in range(self.ui.tabWidget.count()):
            tab = self.ui.tabWidget.widget(i)
            # 检查标签页是否有关联的文件路径
            if hasattr(tab, 'file_path') and tab.file_path == file_path:
                return i
        return -1

    def close_tab(self, tab_widget):
        """
        关闭标签页
        """
        index = self.ui.tabWidget.indexOf(tab_widget)
        if index >= 0:
            self.ui.tabWidget.removeTab(index)
            self.ui.statusBar.setText("标签页已关闭")

    def toggle_checkbox(self, item):
        """切换文件复选框状态"""
        # 获取当前状态
        current_state = self.checkbox_states.get(item, False)
        new_state = not current_state
        self.checkbox_states[item] = new_state
        if new_state:
            checkbox_char = "✓"
            item.setText(1, checkbox_char)
        else:
            checkbox_char = "□"
            item.setText(1, checkbox_char)

        # 更新父文件夹的复选框状态
        parent_item = item.parent()
        if parent_item:
            self.update_parent_folder_state(parent_item)

        # 更新统计信息
        self.update_stats()
        self.apply_checkbox_style()

    def toggle_folder_checkbox(self, folder_item):
        """切换文件夹复选框状态（全选/取消全选文件夹下所有文件）"""
        # 获取当前文件夹状态
        current_state = self.folder_checkbox_states.get(folder_item, False)

        # 切换状态
        new_state = not current_state
        self.folder_checkbox_states[folder_item] = new_state
        if new_state:
            folder_item.setText(1, "✓")
        else:
            folder_item.setText(1, "□")

        # 更新文件夹下所有文件的状态
        self.update_folder_files_state(folder_item, new_state)
        self.update_stats()
        self.apply_checkbox_style()

        folder_name = folder_item.text(0)
        if new_state:
            self.ui.bottom_manager.update_status(f"已全选文件夹 '{folder_name}' 下的所有文件")
        else:
            self.ui.bottom_manager.update_status(f"已取消选择文件夹 '{folder_name}' 下的所有文件")

    def update_folder_files_state(self, folder_item, new_state):
        """更新文件夹下所有文件的状态"""
        file_items = []
        self.get_all_file_items_in_folder(folder_item, file_items)
        for file_item in file_items:
            self.checkbox_states[file_item] = new_state
            if new_state:
                file_item.setText(1, "✓")
            else:
                file_item.setText(1, "□")

    def update_parent_folder_state(self, folder_item):
        """更新父文件夹的复选框状态"""
        file_items = []
        self.get_all_file_items_in_folder(folder_item, file_items)
        if not file_items:
            return

        all_selected = all(self.checkbox_states.get(item, False) for item in file_items)
        none_selected = all(not self.checkbox_states.get(item, False) for item in file_items)
        if all_selected:
            self.folder_checkbox_states[folder_item] = True
            folder_item.setText(1, "✓")
        elif none_selected:
            self.folder_checkbox_states[folder_item] = False
            folder_item.setText(1, "□")
        else:
            self.folder_checkbox_states[folder_item] = False
            folder_item.setText(1, "◐")  # 使用部分选中符号

    def get_all_file_items_in_folder(self, parent_item, file_items):
        """递归获取文件夹下的所有文件项"""
        for i in range(parent_item.childCount()):
            child_item = parent_item.child(i)
            child_type = child_item.data(0, Qt.ItemDataRole.UserRole)

            if child_type == "file":
                file_items.append(child_item)
            elif child_type in ["root", "fixed_root"]:
                self.get_all_file_items_in_folder(child_item, file_items)

    def select_data_directory(self, path_item):
        """选择新的数据计算目录"""
        current_dir = path_item.data(0, Qt.ItemDataRole.UserRole + 1)
        new_dir = QFileDialog.getExistingDirectory(
            None,
            "选择新的数据计算目录",
            current_dir if current_dir else os.path.expanduser("~")
        )

        if new_dir:
            try:
                # 更新固定根节点的数据
                self.fixed_root_item.setData(0, Qt.ItemDataRole.UserRole + 1, new_dir)

                # 更新路径节点的数据
                path_item.setData(0, Qt.ItemDataRole.UserRole + 1, new_dir)
                path_item.setText(0, f"路径: {new_dir}")
                path_item.setToolTip(0, new_dir)
                path_item.setToolTip(1, f"当前目录: {new_dir}")

                self.clear_folder_files(self.fixed_root_item)
                self.scan_folder_files(new_dir, self.fixed_root_item)
                self.folder_checkbox_states[self.fixed_root_item] = False
                self.fixed_root_item.setText(1, "□")
                self.update_stats()
                self.ui.bottom_manager.update_status(f"数据计算目录已更改为: {new_dir}")

            except Exception as e:
                QMessageBox.critical(None, "错误", f"更改数据计算目录时出错:\n{str(e)}")

    def set_data_directory_from_fixed_root(self):
        """从固定根节点设置数据计算目录"""
        current_dir = self.fixed_root_item.data(0, Qt.ItemDataRole.UserRole + 1)
        new_dir = QFileDialog.getExistingDirectory(
            None,
            "选择新的数据计算目录",
            current_dir if current_dir else os.path.expanduser("~")
        )

        if new_dir:
            try:
                self.fixed_root_item.setData(0, Qt.ItemDataRole.UserRole + 1, new_dir)
                path_item = None
                for i in range(self.fixed_root_item.childCount()):
                    child_item = self.fixed_root_item.child(i)
                    if child_item.data(0, Qt.ItemDataRole.UserRole) == "path_select":
                        path_item = child_item
                        break

                if path_item:
                    path_item.setData(0, Qt.ItemDataRole.UserRole + 1, new_dir)
                    path_item.setText(0, f"路径: {new_dir}")
                    path_item.setToolTip(0, new_dir)
                    path_item.setToolTip(1, f"当前目录: {new_dir}")

                self.clear_folder_files(self.fixed_root_item)
                self.scan_folder_files(new_dir, self.fixed_root_item)
                self.folder_checkbox_states[self.fixed_root_item] = False
                self.fixed_root_item.setText(1, "□")
                self.update_stats()
                self.ui.bottom_manager.update_status(f"数据计算目录已更改为: {new_dir}")

            except Exception as e:
                QMessageBox.critical(None, "错误", f"更改数据计算目录时出错:\n{str(e)}")

    def clear_folder_files(self, folder_item):
        """清空文件夹下的文件节点"""
        children_to_remove = []
        for i in range(folder_item.childCount()):
            child_item = folder_item.child(i)
            child_type = child_item.data(0, Qt.ItemDataRole.UserRole)

            if child_type not in ["path", "path_select", "empty", "error"]:
                children_to_remove.append(child_item)

        for child_item in children_to_remove:
            if child_item in self.checkbox_states:
                del self.checkbox_states[child_item]

            folder_item.removeChild(child_item)

    def apply_checkbox_style(self):
        """应用复选框样式"""
        for item in self.checkbox_states:
            item_type = item.data(0, Qt.ItemDataRole.UserRole)
            if item_type == "file":
                if self.checkbox_states[item]:
                    item.setForeground(1, QColor("#3ba4a7"))
                else:
                    item.setForeground(1, QColor("#3ba4a7"))

        for folder_item in self.folder_checkbox_states:
            if self.folder_checkbox_states[folder_item]:
                folder_item.setForeground(1, QColor("#3ba4a7"))
            else:
                folder_item.setForeground(1, QColor("#3ba4a7"))

    def rename_file(self, file_item):
        """重命名文件"""
        try:
            old_file_path = file_item.data(0, Qt.ItemDataRole.UserRole + 1)
            if not old_file_path:
                QMessageBox.warning(None, "警告", "无法获取文件路径")
                return

            # 检查文件是否在数据计算目录下
            if not self.fixed_root_item:
                QMessageBox.warning(None, "警告", "未找到数据计算目录")
                return

            data_dir = self.fixed_root_item.data(0, Qt.ItemDataRole.UserRole + 1)
            if not data_dir:
                QMessageBox.warning(None, "警告", "未找到数据计算目录路径")
                return

            if not old_file_path.startswith(data_dir):
                QMessageBox.warning(None, "警告", "只能重命名数据计算目录下的文件")
                return

            # 获取原文件名和扩展名
            old_file_name = os.path.basename(old_file_path)
            file_ext = os.path.splitext(old_file_path)[1]
            old_file_base = os.path.splitext(old_file_name)[0]

            # 弹出输入对话框，让用户输入新文件名
            new_file_name, ok = QInputDialog.getText(
                None,
                "重命名文件",
                f"请输入新文件名（原文件: {old_file_name}）:",
                QLineEdit.EchoMode.Normal,
                old_file_base
            )

            if not ok or not new_file_name.strip():
                return

            new_file_name = new_file_name.strip()

            # 确保新文件名有扩展名
            if not os.path.splitext(new_file_name)[1]:
                new_file_name = new_file_name + file_ext

            # 构建新的文件路径
            new_file_path = os.path.join(os.path.dirname(old_file_path), new_file_name)

            # 检查新文件是否已存在
            if os.path.exists(new_file_path):
                if old_file_path == new_file_path:
                    return

                reply = QMessageBox.question(
                    None,
                    "确认",
                    f"文件 '{new_file_name}' 已存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            # 执行重命名
            try:
                os.rename(old_file_path, new_file_path)
            except Exception as e:
                QMessageBox.critical(None, "错误", f"重命名失败:\n{str(e)}")
                return

            # 更新树节点
            file_item.setText(0, new_file_name)
            file_item.setData(0, Qt.ItemDataRole.UserRole + 1, new_file_path)
            file_item.setToolTip(0, new_file_path)
            file_item.setToolTip(1, new_file_path)

            # 更新状态栏
            self.ui.bottom_manager.update_status(f"已重命名: {old_file_name} -> {new_file_name}")

            # 更新已打开的标签页
            self.update_tab_file_info(old_file_path, new_file_path)

        except Exception as e:
            QMessageBox.critical(None, "错误", f"重命名文件时出错:\n{str(e)}")

    def update_tab_file_info(self, old_file_path, new_file_path):
        """更新已打开的标签页中的文件路径信息"""
        try:
            for i in range(self.ui.tabWidget.count()):
                tab = self.ui.tabWidget.widget(i)
                if hasattr(tab, 'file_path') and tab.file_path == old_file_path:
                    # 更新标签页的文件路径
                    tab.file_path = new_file_path

                    # 更新标签页标题
                    new_file_name = os.path.basename(new_file_path)
                    self.ui.tabWidget.setTabText(i, new_file_name)

                    # 更新标签页中的路径标签
                    for child in tab.children():
                        if isinstance(child, QLabel) and child.text().startswith("路径:"):
                            child.setText(f"路径: {new_file_path}")
                            break

                    break
        except Exception as e:
            print(f"更新标签页信息时出错: {str(e)}")

    def show_context_menu(self, position):
        """显示右键菜单"""
        item = self.ui.treeWidget.itemAt(position)
        menu = QMenu()
        if item:
            item_type = item.data(0, Qt.ItemDataRole.UserRole)

            # 判断是否为数据计算目录下的文件
            is_in_data_dir = False
            if item_type == "file":
                file_path = item.data(0, Qt.ItemDataRole.UserRole + 1)
                if file_path and self.fixed_root_item:
                    data_dir = self.fixed_root_item.data(0, Qt.ItemDataRole.UserRole + 1)
                    if data_dir and file_path.startswith(data_dir):
                        is_in_data_dir = True

            if item_type in ["root", "fixed_root"]:
                # 文件夹节点
                menu.addAction("全选", lambda: self.select_folder_files(item))
                menu.addAction("取消全选", lambda: self.deselect_folder_files(item))
                menu.addSeparator()

                if item_type == "fixed_root":
                    # 数据计算目录节点 - 添加设置目录选项
                    menu.addAction("设置数据计算目录", lambda: self.set_data_directory_from_fixed_root())
                elif item_type == "root" and item != self.fixed_root_item:
                    menu.addAction("取消导入此文件夹", lambda: self.remove_import(item))

            elif item_type == "file":
                # 文件节点
                checkbox_state = self.checkbox_states.get(item, False)
                if checkbox_state:
                    menu.addAction("取消选择", lambda: self.toggle_checkbox(item))
                else:
                    menu.addAction("选择", lambda: self.toggle_checkbox(item))
                menu.addSeparator()
                menu.addAction("复制文件路径", lambda: self.copy_file_path(item))

                # 只有在数据计算目录下的文件才显示重命名和删除选项
                if is_in_data_dir:
                    menu.addSeparator()
                    menu.addAction("重命名文件", lambda: self.rename_file(item))
                    menu.addAction("删除文件", lambda: self.delete_file(item))

            elif item_type == "path_select":
                # 数据计算目录的路径节点
                menu.addAction("选择新的数据计算目录", lambda: self.select_data_directory(item))

            elif item_type == "path":
                # 普通路径节点
                menu.addAction("文件路径信息")

            else:
                # 其他节点
                menu.addAction("此项不可选择")
        else:
            # 空白区域
            menu.addAction("全选", self.select_all)
            menu.addAction("取消全选", self.deselect_all)
            menu.addSeparator()

            # 检查是否有选中的根节点（排除固定根节点）
            selected_items = self.ui.treeWidget.selectedItems()
            selected_roots = []
            for selected_item in selected_items:
                if selected_item.data(0, Qt.ItemDataRole.UserRole) == "root" and selected_item != self.fixed_root_item:
                    selected_roots.append(selected_item)

            if selected_roots:
                menu.addAction(f"取消导入文件夹 ({len(selected_roots)}个)", self.remove_selected_imports)

            menu.addSeparator()
            menu.addAction("清空所有文件夹", self.clear_all)

        # 显示菜单
        menu.exec(self.ui.treeWidget.viewport().mapToGlobal(position))

    def delete_file(self, file_item):
        """删除数据计算目录下的文件"""
        try:
            file_path = file_item.data(0, Qt.ItemDataRole.UserRole + 1)
            if not file_path:
                QMessageBox.warning(None, "警告", "无法获取文件路径")
                return

            # 确认文件是否在数据计算目录下
            if not self.fixed_root_item:
                QMessageBox.warning(None, "警告", "未找到数据计算目录")
                return

            data_dir = self.fixed_root_item.data(0, Qt.ItemDataRole.UserRole + 1)
            if not data_dir:
                QMessageBox.warning(None, "警告", "未找到数据计算目录路径")
                return

            if not file_path.startswith(data_dir):
                QMessageBox.warning(None, "警告", "只能删除数据计算目录下的文件")
                return

            # 确认删除
            file_name = os.path.basename(file_path)
            reply = QMessageBox.question(
                None,
                "确认删除",
                f"确定要删除文件 '{file_name}' 吗？\n\n"
                f"路径: {file_path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # 删除文件
            try:
                os.remove(file_path)
                self.ui.bottom_manager.update_status(f"已删除文件: {file_name}")
            except Exception as e:
                QMessageBox.critical(None, "错误", f"删除文件失败:\n{str(e)}")
                return

            # 从树中移除节点
            parent_item = file_item.parent()
            if parent_item:
                parent_item.removeChild(file_item)

            # 清理状态字典
            if file_item in self.checkbox_states:
                del self.checkbox_states[file_item]

            # 更新父文件夹状态
            if parent_item:
                self.update_parent_folder_state(parent_item)

            # 更新统计信息
            self.update_stats()

            # 如果有对应的标签页，关闭它
            self.close_tab_if_open(file_path)

        except Exception as e:
            QMessageBox.critical(None, "错误", f"删除文件时出错:\n{str(e)}")

    def close_tab_if_open(self, file_path):
        """如果文件对应的标签页已打开，关闭它"""
        try:
            for i in range(self.ui.tabWidget.count()):
                tab = self.ui.tabWidget.widget(i)
                if hasattr(tab, 'file_path') and tab.file_path == file_path:
                    self.ui.tabWidget.removeTab(i)
                    self.ui.statusBar.setText("已关闭已删除文件的标签页")
                    break
        except Exception as e:
            print(f"关闭标签页时出错: {str(e)}")

    def select_folder_files(self, folder_item):
        """选择指定文件夹下的所有文件"""
        self.folder_checkbox_states[folder_item] = True
        folder_item.setText(1, "✓")
        self.update_folder_files_state(folder_item, True)
        self.update_stats()
        self.ui.bottom_manager.update_status("已选择此文件夹下所有文件")
        self.ui.statusBar.setText("")

    def deselect_folder_files(self, folder_item):
        """取消选择指定文件夹下的所有文件"""
        self.folder_checkbox_states[folder_item] = False
        folder_item.setText(1, "□")
        self.update_folder_files_state(folder_item, False)
        self.update_stats()
        self.ui.bottom_manager.update_status("已取消选择此文件夹下所有文件")

    def copy_file_path(self, file_item):
        """复制文件路径到剪贴板"""
        file_path = file_item.data(0, Qt.ItemDataRole.UserRole + 1)
        if file_path:
            clipboard = QApplication.clipboard()
            clipboard.setText(file_path)
            self.ui.bottom_manager.update_status(f"已复制文件路径: {file_path}")

    def remove_import(self, root_item):
        """移除单个导入的文件夹"""
        folder_name = None
        folder_path = None

        for path, info in list(self.imported_roots.items()):
            if info['item'] == root_item:
                folder_name = info['name']
                folder_path = path
                break

        if not folder_name:
            return

        reply = QMessageBox.question(
            None,
            "确认",
            f"确定要取消导入 '{folder_name}' 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.remove_all_children(root_item)
            index = self.ui.treeWidget.indexOfTopLevelItem(root_item)
            if index >= 0:
                self.ui.treeWidget.takeTopLevelItem(index)

            if folder_path in self.imported_roots:
                del self.imported_roots[folder_path]

            if root_item in self.folder_checkbox_states:
                del self.folder_checkbox_states[root_item]

            self.update_stats()
            self.ui.bottom_manager.update_status(f"已取消导入: {folder_name}")

    def remove_all_children(self, parent_item):
        """递归删除所有子节点的复选框状态"""
        for i in range(parent_item.childCount()):
            child_item = parent_item.child(i)

            if child_item in self.checkbox_states:
                del self.checkbox_states[child_item]

            if child_item in self.folder_checkbox_states:
                del self.folder_checkbox_states[child_item]

            self.remove_all_children(child_item)

    def remove_selected_imports(self):
        """取消导入所有选中的文件夹"""
        selected_roots = []
        selected_items = self.ui.treeWidget.selectedItems()

        for item in selected_items:
            if item.data(0, Qt.ItemDataRole.UserRole) == "root" and item != self.fixed_root_item:
                selected_roots.append(item)

        if not selected_roots:
            QMessageBox.information(None, "提示", "请先选中要取消导入的文件夹")
            return

        reply = QMessageBox.question(
            None,
            "确认",
            f"确定要取消导入 {len(selected_roots)} 个选中的文件夹吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for root_item in selected_roots:
                self.remove_all_children(root_item)

                index = self.ui.treeWidget.indexOfTopLevelItem(root_item)
                if index >= 0:
                    self.ui.treeWidget.takeTopLevelItem(index)

                for path, info in list(self.imported_roots.items()):
                    if info['item'] == root_item:
                        del self.imported_roots[path]
                        break

                if root_item in self.folder_checkbox_states:
                    del self.folder_checkbox_states[root_item]

            self.update_stats()
            self.ui.bottom_manager.update_status(f"已取消导入 {len(selected_roots)} 个文件夹")

    def select_all(self):
        """全选所有文件"""
        for folder_item in self.folder_checkbox_states:
            self.folder_checkbox_states[folder_item] = True
            folder_item.setText(1, "✓")

        for file_item in self.checkbox_states:
            self.checkbox_states[file_item] = True
            file_item.setText(1, "✓")

        self.update_stats()
        self.ui.bottom_manager.update_status(f"已全选所有文件")

    def deselect_all(self):
        """取消全选所有项目"""
        for folder_item in self.folder_checkbox_states:
            self.folder_checkbox_states[folder_item] = False
            folder_item.setText(1, "□")

        for file_item in self.checkbox_states:
            self.checkbox_states[file_item] = False
            file_item.setText(1, "□")

        self.update_stats()
        self.ui.bottom_manager.update_status("已取消全选")

    def clear_all(self):
        """清空所有导入的文件夹（保留固定根节点）"""
        root_items_to_remove = []
        for i in range(self.ui.treeWidget.topLevelItemCount()):
            item = self.ui.treeWidget.topLevelItem(i)
            if item != self.fixed_root_item and item.data(0, Qt.ItemDataRole.UserRole) == "root":
                root_items_to_remove.append(item)

        if not root_items_to_remove:
            QMessageBox.information(None, "提示", "没有可清空的文件夹")
            return

        reply = QMessageBox.question(
            None,
            "确认",
            f"确定要清空所有导入的文件夹吗？（共{len(root_items_to_remove)}个）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for root_item in root_items_to_remove:
                self.remove_all_children(root_item)
                index = self.ui.treeWidget.indexOfTopLevelItem(root_item)
                if index >= 0:
                    self.ui.treeWidget.takeTopLevelItem(index)

                for path, info in list(self.imported_roots.items()):
                    if info['item'] == root_item:
                        del self.imported_roots[path]
                        break

            self.update_stats()
            self.ui.bottom_manager.update_status(f"已清空 {len(root_items_to_remove)} 个文件夹")

    def update_stats(self):
        """更新统计信息"""
        folder_count = len(self.imported_roots)
        selected_count = 0
        for item, is_selected in self.checkbox_states.items():
            if is_selected and item.data(0, Qt.ItemDataRole.UserRole) == "file":
                selected_count += 1

        self.ui.bottom_manager.update_info(f"已导入:{folder_count}个文件夹|选中:{selected_count}个文件")

    def get_selected_files(self):
        """获取所有选中的文件路径"""
        selected_files = []
        for item, is_selected in self.checkbox_states.items():
            if is_selected and item.data(0, Qt.ItemDataRole.UserRole) == "file":
                file_path = item.data(0, Qt.ItemDataRole.UserRole + 1)
                if file_path:
                    selected_files.append(file_path)
        return selected_files

    def get_imported_folders(self):
        """获取所有导入的文件夹信息"""
        return self.imported_roots.copy()

    def get_data_directory_path(self):
        """获取数据计算目录的路径数据"""
        if self.fixed_root_item:
            dir_path = self.fixed_root_item.data(0, Qt.ItemDataRole.UserRole + 1)
            return dir_path
        return None

    def refresh_directory(self, directory_path=None):
        """刷新指定目录的文件列表

        Args:
            directory_path: 要刷新的目录路径。如果为None，则刷新数据计算目录
        """
        try:
            if directory_path is None:
                # 刷新数据计算目录
                directory_path = self.get_data_directory_path()
                target_item = self.fixed_root_item
                item_type = "fixed_root"
            else:
                # 查找对应的根节点
                target_item = None
                item_type = None

                # 在导入的根节点中查找
                for path, info in self.imported_roots.items():
                    if path == directory_path:
                        target_item = info['item']
                        item_type = "root"
                        break

                # 如果在导入节点中没找到，检查是否是数据计算目录
                if target_item is None and self.fixed_root_item:
                    fixed_root_path = self.fixed_root_item.data(0, Qt.ItemDataRole.UserRole + 1)
                    if fixed_root_path == directory_path:
                        target_item = self.fixed_root_item
                        item_type = "fixed_root"

            if target_item is None:
                print(f"未找到目录对应的节点: {directory_path}")
                return False

            # 验证目录是否存在
            if not os.path.exists(directory_path):
                QMessageBox.warning(None, "警告", f"目录不存在:\n{directory_path}")
                return False

            # 清空目录下的文件节点
            self.clear_folder_files(target_item)

            # 重新扫描文件
            self.scan_folder_files(directory_path, target_item)

            # 重置文件夹的复选框状态
            if target_item in self.folder_checkbox_states:
                self.folder_checkbox_states[target_item] = False

            # 更新复选框文本
            target_item.setText(1, "□")

            # 更新统计信息
            self.update_stats()

            # 更新状态栏
            dir_name = os.path.basename(directory_path)
            self.ui.bottom_manager.update_status(f"已刷新目录: {dir_name}")

            print(f"成功刷新目录: {directory_path}")
            return True

        except Exception as e:
            print(f"刷新目录时出错: {str(e)}")
            self.ui.bottom_manager.update_status(f"刷新目录失败: {str(e)}")
            return False

    def refresh_data_directory(self):
        """刷新数据计算目录（专用方法）"""
        return self.refresh_directory(None)