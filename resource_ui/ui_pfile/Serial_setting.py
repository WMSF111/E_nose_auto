import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QScrollArea
import global_var as glo_var


class SerialSetting_Init(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("传感器设置")
        self.setGeometry(100, 100, 300, 300)
        self.setFixedSize(300, 300)  # 设置窗口大小固定

        # 创建一个中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建一个垂直布局
        self.layout = QVBoxLayout(self.central_widget)
        # 创建一个按钮用于添加标签
        self.already_button = QPushButton("确定传感器", self.central_widget)
        self.already_button.clicked.connect(self.update_sensors)
        self.layout.addWidget(self.already_button)

        # 创建一个垂直布局
        Hlayout = QHBoxLayout()
        # 创建一个输入框用于输入标签内容
        self.input_field = QLineEdit(self.central_widget)
        self.input_field.setPlaceholderText("请输入传感器标签")
        Hlayout.addWidget(self.input_field)

        # 创建一个按钮用于添加标签
        self.add_button = QPushButton("添加传感器", self.central_widget)
        self.add_button.clicked.connect(self.add_label)
        Hlayout.addWidget(self.add_button)
        self.layout.addLayout(Hlayout)


        # 创建一个滚动区域用于存储标签
        self.scroll_area = QScrollArea(self.central_widget)
        self.scroll_area.setWidgetResizable(True)  # 设置滚动区域的大小可调整
        self.layout.addWidget(self.scroll_area)

        # 创建一个容器用于存储标签
        self.label_container = QWidget()
        self.label_layout = QVBoxLayout(self.label_container)
        self.label_layout.setSpacing(5)  # 设置标签之间的间距为5像素
        self.scroll_area.setWidget(self.label_container)  # 将标签容器设置为滚动区域的部件

        # 初始化默认传感器列表
        self.initialize_default_sensors()

    def initialize_default_sensors(self):

        # 将默认传感器名称添加到 self.sensors_Item
        self.sensors_Item = []
        for sensor_name in glo_var.sensors:
            self.add_label(sensor_name)

    def add_label(self, text=None):
        # flag = 0
        # 如果是通过按钮添加，则从输入框获取内容
        if text is False :
            # flag = 1
            text = self.input_field.text()
            if len(text) == 0:
                return;

        if text:  # 如果输入框不为空
            # 创建一个新的 QHBoxLayout 用于放置序号和标签
            label_layout = QHBoxLayout()

            # 创建序号标签
            index_label = QLabel(str(len(self.sensors_Item) + 1) + ".", self.label_container)
            index_label.setFixedSize(30, 20)  # 设置宽度为30像素，高度为20像素
            label_layout.addWidget(index_label)

            # 创建内容标签
            new_label = QLineEdit(text, self.label_container)
            new_label.setStyleSheet("border: 1px solid white; padding: 5px;")  # 添加一些样式
            label_layout.addWidget(new_label)

            # 创建删除按钮
            delete_button = QPushButton("删除", self.label_container)
            delete_button.clicked.connect(lambda checked, label=new_label: self.delete_label(label))
            delete_button.setFixedSize(80, 30)  # 设置宽度为80像素，高度为30像素
            label_layout.addWidget(delete_button)

            # 将布局添加到容器中
            self.label_layout.addLayout(label_layout)

            # 将新标签添加到列表中
            self.sensors_Item.append((index_label, new_label, delete_button))

            # 清空输入框（如果是用户输入添加的）
            if text is None:
                self.input_field.clear()

            # 更新所有标签的序号
            self.update_label_indices()
            # if flag == 1:
            #     self.update_sensors()


    def delete_label(self, label_to_delete):
        # 找到要删除的标签
        for index, (index_label, label, delete_button) in enumerate(self.sensors_Item):
            if label == label_to_delete:
                # 删除标签、序号标签和删除按钮
                label.deleteLater()
                index_label.deleteLater()
                delete_button.deleteLater()
                # 从列表中移除
                del self.sensors_Item[index]
                break

        # 更新所有标签的序号
        self.update_label_indices()

    def update_label_indices(self):
        # 更新所有标签的序号
        for index, (index_label, new_label, _) in enumerate(self.sensors_Item):
            index_label.setText(str(index + 1) + ".")


    def update_sensors(self):
        glo_var.sensors.clear()
        # 更新所有标签的序号
        for index, (index_label, new_label, _) in enumerate(self.sensors_Item):
            glo_var.sensors.append(new_label.text())
        print(glo_var.sensors)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_window = SerialSetting_Init()
#     main_window.show()
#     sys.exit(app.exec())