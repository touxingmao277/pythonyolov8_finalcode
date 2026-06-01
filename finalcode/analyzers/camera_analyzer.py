import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..exercise.engine import ExerciseAnalyzer, AnalysisResult

class CameraAnalyzer:
    """摄像头实时动作分析器

    从摄像头读取视频流并进行实时动作分析，支持骨骼线绘制、
    角度标注和状态信息显示。按 'q' 键退出。
    """

    def __init__(self, analyzer: ExerciseAnalyzer,
                 camera_id: int = 0, width: int = 640,
                 height: int = 480) -> None:
        """初始化摄像头分析器

        参数:
            analyzer: 动作分析器实例
            camera_id: 摄像头设备ID，默认为0（第一个摄像头）
            width: 视频帧宽度
            height: 视频帧高度
        """
        self.analyzer = analyzer
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self._display_msg = ""
        self._pending_msg = ""
        self._msg_stable_count = 0

    def start(self, window_name: str = "Pose Analysis") -> None:
        """启动摄像头分析

        参数:
            window_name: 窗口名称
        """
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise ValueError(f"无法连接摄像头: {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                break

            pose_result = self.analyzer.pose_detector.detect_pose(frame)
            analysis_result = self.analyzer.analyze(frame, pose_result)

            if pose_result:
                frame = self.analyzer.pose_detector.draw_skeleton(frame, pose_result)

            frame = self._draw_info(frame, analysis_result)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

        cv2.destroyAllWindows()

    def stop(self) -> None:
        """停止摄像头分析，释放资源"""
        self.running = False
        if self.cap:
            self.cap.release()

    @staticmethod
    def _draw_text(draw, xy, text, font, fill, shadow_offset=2):
        """带阴影的文字，在任何背景下都清晰可见"""
        x, y = xy
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=fill)

    def _draw_info(self, frame: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """在图像上绘制分析信息

        参数:
            frame: 输入图像
            result: 分析结果

        返回:
            绘制了分析信息的图像
        """
        frame_copy = frame.copy()

        try:
            font_count = ImageFont.truetype("msyh.ttc", 32)
            font = ImageFont.truetype("msyh.ttc", 26)
            font_small = ImageFont.truetype("msyh.ttc", 18)
        except:
            font_count = font = font_small = ImageFont.load_default()

        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)

        if result.is_detected:
            for angle in result.angles:
                text = f"{angle.value:.0f}°"
                if hasattr(angle, 'joint_pos') and angle.joint_pos is not None:
                    x, y = angle.joint_pos
                    color = (0, 255, 80) if angle.is_standard else (50, 180, 255)
                    self._draw_text(draw, (int(x) + 10, int(y) - 20), text, font, color)
                    self._draw_text(draw, (int(x) + 10, int(y)), angle.name, font_small, (255, 255, 255))

            stats = self.analyzer.get_statistics()
            self._draw_text(draw, (12, 12), f"次数: {stats.total_count}",
                           font_count, fill=(0, 230, 255))

            if result.feedback.message:
                self._update_display_msg(result.feedback.message)
                msg_color = (0, 255, 80) if result.feedback.is_standard else (255, 200, 40)
                self._draw_text(draw, (12, 50), self._display_msg, font, fill=msg_color)
        else:
            self._update_display_msg(result.feedback.message)
            self._draw_text(draw, (12, 12), self._display_msg, font_count, fill=(50, 130, 255))

        frame_copy = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return frame_copy

    def _update_display_msg(self, new_msg: str) -> None:
        """消息需稳定 5 帧才更新显示，防止文字跳动"""
        if new_msg == self._pending_msg:
            self._msg_stable_count += 1
            if self._msg_stable_count >= 5:
                self._display_msg = self._pending_msg
                self._msg_stable_count = 0
        else:
            self._pending_msg = new_msg
            self._msg_stable_count = 0
