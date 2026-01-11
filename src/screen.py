import sys,torch,cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QPainter, QPen, QImage, QColor
from PyQt6.QtCore import Qt, QPoint

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280) # 画板放大10倍，方便书写
        self.setCursor(Qt.CursorShape.CrossCursor) #
        self.image = QImage(self.size(), QImage.Format.Format_Grayscale8)
        self.clear_canvas()
        self.last_point = QPoint()

    def clear_canvas(self):# 清空画板
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def paintEvent(self, event):# 绑定绘制事件
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def mousePressEvent(self, event):#鼠标按下事件
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):#鼠标移动事件
        if event.buttons() & Qt.MouseButton.LeftButton:
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing) # 抗锯齿
            painter.setPen(QPen(Qt.GlobalColor.black, 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def get_image(self):
        #想法是，先将Qimage格式通过指针转为numpy格式的三维张量，随后进行二值化，再找到包含信息的最小外接矩形，
        # 按照比例将长边缩放到20像素，短边也一同缩放，随后将周边部分填充。得到最后的张量信息。
        ptr = self.image.bits()
        ptr.setsize(self.image.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape((self.image.height(), self.image.width()))
        _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(binary) #颜色反转
        pts = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(pts)
        roi = mask[y:y + h, x:x + w]
        side = max(w, h)
        scale = 20 / side
        new_w, new_h = int(w * scale), int(h * scale)
        resized_digit = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        final_img = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        final_img[start_y: start_y + new_h, start_x: start_x + new_w] = resized_digit #居中放置
        return final_img

class DigitRecognizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于CNN的数字识别")
        self.setStyleSheet("background-color: #1e1e1e; color: white;")  # 深色主题
        self.cnn = torch.load("../models/mnist_model.pkl", weights_only=False)
        self.cnn = self.cnn.cuda()
        # 主布局
        main_layout = QHBoxLayout()

        # 画板区
        left_layout = QVBoxLayout()
        self.canvas = Canvas()
        self.canvas.setStyleSheet("border: 5px solid #3d3d3d; border-radius: 10px;")
        left_layout.addWidget(QLabel("在下方书写数字"))
        left_layout.addWidget(self.canvas)

        btn_clear = QPushButton("清空重写")
        btn_clear.setStyleSheet("background-color: #d9534f; padding: 10px; border-radius: 5px;")
        btn_clear.clicked.connect(self.canvas.clear_canvas)
        left_layout.addWidget(btn_clear)

        # 结果展示区
        right_layout = QVBoxLayout()
        self.result_label = QLabel("?")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 100px; color: #5cb85c; font-weight: bold;")

        btn_predict = QPushButton("识别数字")
        btn_predict.setStyleSheet("background-color: #5bc0de; padding: 15px; font-size: 18px; border-radius: 5px;")
        btn_predict.clicked.connect(self.recognize)  # 点击触发识别

        right_layout.addWidget(QLabel("预测结果"))
        right_layout.addWidget(self.result_label)
        right_layout.addStretch()  # 弹簧挤压
        right_layout.addWidget(btn_predict)

        main_layout.addLayout(left_layout)
        main_layout.addSpacing(30)  # 间距
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def recognize(self):
        img = self.canvas.get_image()
        img = torch.FloatTensor(img) / 255.0
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.cuda()

        output = self.cnn(img)
        _, pred = output.max(1)

        self.result_label.setText("{}".format(pred.item()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognizer()
    window.show()
    sys.exit(app.exec())