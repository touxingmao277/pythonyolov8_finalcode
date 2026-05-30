import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..exercise.engine import ExerciseAnalyzer, AnalysisResult, ExerciseStatistics

class VideoAnalyzer:
    def __init__(self, analyzer: ExerciseAnalyzer) -> None:
        self.analyzer = analyzer
        # 显示层去抖动：消息需稳定 N 帧才更新
        self._display_msg = ""
        self._pending_msg = ""
        self._msg_stable_count = 0

    def analyze_video(self, video_path: str, output_path: str = None,
                     show_preview: bool = True) -> ExerciseStatistics:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            pose_result = self.analyzer.pose_detector.detect_pose(frame)
            analysis_result = self.analyzer.analyze(frame, pose_result)

            if pose_result:
                frame = self.analyzer.pose_detector.draw_skeleton(frame, pose_result)

            frame = self._draw_info(frame, analysis_result)

            if out:
                out.write(frame)

            if show_preview:
                cv2.imshow('Video Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        return self.analyzer.get_statistics()

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        pose_result = self.analyzer.pose_detector.detect_pose(frame)
        return self.analyzer.analyze(frame, pose_result)

    @staticmethod
    def _draw_text(draw, xy, text, font, fill, shadow_offset=2):
        """带阴影的文字，在任何背景下都清晰可见"""
        x, y = xy
        # 先画黑色阴影
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))
        # 再画主体颜色
        draw.text((x, y), text, font=font, fill=fill)

    def _draw_info(self, frame: np.ndarray, result: AnalysisResult) -> np.ndarray:
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
            # 关节角度标注
            for angle in result.angles:
                text = f"{angle.value:.0f}°"
                if hasattr(angle, 'joint_pos') and angle.joint_pos is not None:
                    x, y = angle.joint_pos
                    color = (0, 255, 80) if angle.is_standard else (50, 180, 255)
                    self._draw_text(draw, (int(x) + 10, int(y) - 20), text, font, color)
                    self._draw_text(draw, (int(x) + 10, int(y)), angle.name, font_small, (255, 255, 255))

            stats = self.analyzer.get_statistics()
            # 次数 — 金色大字
            self._draw_text(draw, (12, 12), f"次数: {stats.total_count}",
                           font_count, fill=(0, 230, 255))

            # 状态消息
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
