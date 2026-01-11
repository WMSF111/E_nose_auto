import pandas as pd
from PyQt6.QtGui import QFont
from PySide6.QtWidgets import *

class CSVTableWidget(QWidget):
    """显示csv文件"""
    def __init__(self, file_path):
        super().__init__()
        layout = QVBoxLayout()
        # 创建TableWidget
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)  # 只读模式
        layout.addWidget(self.table_widget)
        self.setLayout(layout)

        # 使用样式表设置字体
        # self.table_widget.setStyleSheet("""
        #     QTableWidget {
        #         font-family: "Microsoft YaHei";
        #         font-size: 9px;
        #         font-weight: normal;
        #     }
        #     QHeaderView::section {
        #         font-family: "SimHei";
        #         font-size: 11px;
        #         font-weight: normal;
        #         color: white;
        #     }
        # """)

        try:
            df = pd.read_csv(file_path)
            rows, cols = df.shape   # 行数 列数
            # 设置表格的行列数
            self.table_widget.setRowCount(rows)
            self.table_widget.setColumnCount(cols)
            # 设置表头
            headers = df.columns.tolist()
            self.table_widget.setHorizontalHeaderLabels(headers)
            # 填充数据
            for row in range(rows):
                for col in range(cols):
                    item_value = str(df.iloc[row, col])
                    item = QTableWidgetItem(item_value)
                    self.table_widget.setItem(row, col, item)

            # 调整列宽适应内容
            self.table_widget.resizeColumnsToContents()
        except Exception as e:
            QMessageBox.warning(None, "错误", f"加载CSV文件失败: {str(e)}")