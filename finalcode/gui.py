import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QTextEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finalcode.pose_detector import PoseDetector
from finalcode.angle_utils import AngleCalculator
from finalcode.exercise.squat_analyzer import SquatAnalyzer
from finalcode.exercise.pushup_analyzer import PushUpAnalyzer

class PoseAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人体动作分析系统")
        self.setGeometry(100, 100, 900, 700)
        
        self.pose_detector = PoseDetector()
        self.angle_calculator = AngleCalculator()
        self.current_analyzer = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        left_layout.addWidget(self.video_label)
        
        control_layout = QHBoxLayout()
        
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems(["深蹲", "俯卧撑"])
        control_layout.addWidget(QLabel("运动类型:"))
        control_layout.addWidget(self.exercise_combo)
        
        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.start_analysis)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(control_layout)
        main_layout.addLayout(left_layout)
        
        right_layout = QVBoxLayout()
        
        self.angle_group = QWidget()
        angle_layout = QVBoxLayout(self.angle_group)
        self.angle_labels = []
        for i in range(2):
            lbl = QLabel()
            self.angle_labels.append(lbl)
            angle_layout.addWidget(lbl)
        right_layout.addWidget(self.angle_group)
        
        self.stats_group = QWidget()
        stats_layout = QVBoxLayout(self.stats_group)
        
        self.count_label = QLabel("动作次数: 0")
        stats_layout.addWidget(self.count_label)
        
        right_layout.addWidget(self.stats_group)
        
        self.feedback_text = QTextEdit()
        self.feedback_text.setReadOnly(True)
        self.feedback_text.setFixedHeight(150)
        right_layout.addWidget(self.feedback_text)
        
        main_layout.addLayout(right_layout)
    
    def start_analysis(self):
        exercise = self.exercise_combo.currentText()
        
        if exercise == "深蹲":
            self.current_analyzer = SquatAnalyzer(self.pose_detector, self.angle_calculator)
        else:
            self.current_analyzer = PushUpAnalyzer(self.pose_detector, self.angle_calculator)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.feedback_text.append("错误: 无法连接摄像头")
            return
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.exercise_combo.setEnabled(False)
        self.timer.start(30)
    
    def stop_analysis(self):
        self.timer.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.video_label.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.exercise_combo.setEnabled(True)
        
        if self.current_analyzer:
            stats = self.current_analyzer.get_statistics()
            self.feedback_text.append(f"\n分析结束 - 总次数: {stats.total_count}, 标准次数: {stats.standard_count}")
    
    def update_frame(self):
        if not self.cap or not self.current_analyzer:
            return
        
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        pose_result = self.pose_detector.detect_pose(frame)
        analysis_result = self.current_analyzer.analyze(frame, pose_result)
        
        if pose_result:
            frame = self.pose_detector.draw_skeleton(frame, pose_result)
        
        frame = self.draw_info(frame, analysis_result)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))
        
        self.update_stats()
        self.update_feedback(analysis_result)
    
    @staticmethod
    def _draw_text(draw, xy, text, font, fill, shadow_offset=2):
        """带阴影的文字，在任何背景下都清晰可见"""
        x, y = xy
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=fill)

    def draw_info(self, frame: np.ndarray, result) -> np.ndarray:
        frame_copy = frame.copy()

        try:
            font = ImageFont.truetype("msyh.ttc", 22)
        except Exception:
            font = ImageFont.load_default()

        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)

        if result.is_detected:
            y_offset = 30
            for angle in result.angles:
                color = (0, 255, 80) if angle.is_standard else (50, 130, 255)
                text = f"{angle.name}: {angle.value:.1f}°"
                self._draw_text(draw, (10, y_offset), text, font, color)
                y_offset += 25

            if result.feedback.message:
                msg_color = (0, 255, 80) if result.feedback.is_standard else (255, 200, 40)
                self._draw_text(draw, (10, y_offset), result.feedback.message, font, msg_color)
        else:
            self._draw_text(draw, (10, 30), result.feedback.message, font, (50, 130, 255))

        frame_copy = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return frame_copy
    
    def update_stats(self):
        if not self.current_analyzer:
            return

        stats = self.current_analyzer.get_statistics()
        self.count_label.setText(f"动作次数: {stats.total_count}")
    
    def update_feedback(self, result):
        if not result.is_detected:
            self._set_feedback_stable(result.feedback.message)
            return

        feedback_text = f"状态: {result.feedback.message}\n\n"

        if result.feedback.issues:
            feedback_text += "问题:\n"
            for issue in result.feedback.issues:
                feedback_text += f"- {issue}\n"

        if result.feedback.suggestions:
            feedback_text += "\n建议:\n"
            for suggestion in result.feedback.suggestions:
                feedback_text += f"- {suggestion}\n"

        self._set_feedback_stable(feedback_text)

    def _set_feedback_stable(self, text: str) -> None:
        """消息需稳定 5 帧才更新，防止文字跳动"""
        if not hasattr(self, '_fb_pending'):
            self._fb_pending = ""
            self._fb_count = 0
        if text == self._fb_pending:
            self._fb_count += 1
            if self._fb_count >= 5:
                self.feedback_text.setText(self._fb_pending)
                self._fb_count = 0
        else:
            self._fb_pending = text
            self._fb_count = 0
    
    def closeEvent(self, event):
        self.stop_analysis()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = PoseAnalysisGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()