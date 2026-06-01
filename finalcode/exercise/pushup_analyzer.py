from collections import deque
from enum import Enum
import numpy as np
from ..pose_detector import PoseDetector, PoseResult
from ..angle_utils import AngleCalculator
from .engine import ExerciseAnalyzer, AngleReading, Feedback, AnalysisResult

class Phase(Enum):
    """俯卧撑动作阶段枚举"""
    UP = 0          # 撑起状态
    DESCENDING = 1  # 下降中
    BOTTOM = 2      # 底部
    ASCENDING = 3   # 推起中

class PushUpAnalyzer(ExerciseAnalyzer):
    """俯卧撑动作分析器

    通过分析人体关键点角度变化，检测俯卧撑动作的完成情况。
    支持自适应校准（根据用户初始姿势调整判定阈值）、
    三帧移动平均平滑处理、以及阶段状态机逻辑。
    """

    def __init__(self, pose_detector: PoseDetector,
                 angle_calculator: AngleCalculator) -> None:
        """初始化俯卧撑分析器

        参数:
            pose_detector: 姿态检测器实例
            angle_calculator: 角度计算器实例
        """
        super().__init__(pose_detector, angle_calculator)

        self.config = {
            "elbow_angle_bottom_min": 60,
            "elbow_angle_bottom_max": 100,
            "shoulder_angle_bottom_min": 160,
            "shoulder_angle_bottom_max": 180,
            "elbow_angle_up_min": 150,
            "elbow_angle_up_max": 180,
            "shoulder_angle_up_min": 165,
            "shoulder_angle_up_max": 180,
            "elbow_angle_transition_min": 55,
            "elbow_angle_transition_max": 165,
            "shoulder_angle_transition_min": 155,
            "shoulder_angle_transition_max": 180,
        }

        self.current_phase = Phase.UP
        self.previous_elbow_angle = 180.0
        self.frame_number = 0

        self._calibrated = False
        self._calibration_frames = 30
        self._up_elbow_samples = []

        self.descend_threshold = 140
        self.bottom_threshold = 95
        self.up_threshold = 155

        self._elbow_buffer = deque(maxlen=3)
        self._shoulder_buffer = deque(maxlen=3)

        self.phase_debounce = 5
        self._pending_phase = None
        self._pending_count = 0

        self._hit_bottom = False
        self._rep_start_frame = 0
        self._min_rep_frames = 15

    def get_exercise_name(self) -> str:
        return "俯卧撑"

    def analyze(self, frame: np.ndarray, results: PoseResult) -> AnalysisResult:
        """分析单帧图像中的俯卧撑动作

        参数:
            frame: 输入图像
            results: 姿态检测结果

        返回:
            AnalysisResult 包含肘部角度、身体角度、阶段状态和反馈信息
        """
        self.frame_number += 1

        if results is None or len(results.keypoints) < 3:
            return AnalysisResult(
                is_detected=False, angles=[],
                feedback=Feedback(is_standard=False, message="未检测到人体",
                                  issues=[], suggestions=[]),
                timestamp=0.0, frame_number=self.frame_number
            )

        shoulder = self.pose_detector.get_keypoint_with_fallback(5, 6, results)
        elbow = self.pose_detector.get_keypoint_with_fallback(7, 8, results)
        wrist = self.pose_detector.get_keypoint_with_fallback(9, 10, results)
        hip = self.pose_detector.get_keypoint_with_fallback(11, 12, results)
        ankle = self.pose_detector.get_keypoint_with_fallback(15, 16, results)

        angles = []
        issues = []
        suggestions = []
        elbow_angle = None
        body_angle = None

        if shoulder is not None and elbow is not None and wrist is not None:
            raw = self.angle_calculator.calculate_angle(shoulder, elbow, wrist)
            self._elbow_buffer.append(raw)
            elbow_angle = np.mean(self._elbow_buffer) if self._elbow_buffer else raw

            elbow_min, elbow_max = self._get_standard_range("elbow")
            elbow_is_standard = self.angle_calculator.is_angle_in_range(
                elbow_angle, elbow_min, elbow_max)

            angles.append(AngleReading(
                "肘部角度", elbow_angle, elbow_is_standard,
                elbow_min, elbow_max, joint_pos=elbow))

            if not elbow_is_standard:
                issues.append(f"肘部: {elbow_angle:.0f}° (标准 {elbow_min}-{elbow_max}°)")
                if self.current_phase == Phase.UP:
                    suggestions.append("伸直手臂")
                elif self.current_phase == Phase.BOTTOM:
                    suggestions.append("屈肘至约90度")

        if shoulder is not None and hip is not None and ankle is not None:
            raw_body = self.angle_calculator.calculate_angle(shoulder, hip, ankle)
            self._shoulder_buffer.append(raw_body)
            body_angle = np.mean(self._shoulder_buffer) if self._shoulder_buffer else raw_body

            body_min, body_max = self._get_standard_range("shoulder")
            body_is_standard = self.angle_calculator.is_angle_in_range(
                body_angle, body_min, body_max)

            angles.append(AngleReading(
                "身体角度", body_angle, body_is_standard,
                body_min, body_max, joint_pos=hip))

            if not body_is_standard:
                issues.append(f"身体: {body_angle:.0f}° (标准 {body_min}-{body_max}°)")
                if body_angle < body_min:
                    suggestions.append("不要塌腰，收紧核心")
                else:
                    suggestions.append("不要拱背，保持直线")

        if not angles:
            return AnalysisResult(
                is_detected=True, angles=[],
                feedback=Feedback(is_standard=False, message="关键点不足",
                                  issues=[], suggestions=[]),
                timestamp=results.timestamp if results else 0.0,
                frame_number=self.frame_number
            )

        if not self._calibrated and elbow_angle is not None:
            self._calibrate(elbow_angle)

        if elbow_angle is not None:
            self._update_phase(elbow_angle)

        angle_standards = [a.is_standard for a in angles]
        is_standard = all(angle_standards) if angle_standards else False

        phase_name = self._phase_name()
        message = phase_name

        feedback = Feedback(
            is_standard=is_standard, message=message,
            issues=issues, suggestions=suggestions)

        self.total_frames += 1
        if is_standard:
            self.standard_frames += 1

        return AnalysisResult(
            is_detected=True, angles=angles, feedback=feedback,
            timestamp=results.timestamp, frame_number=self.frame_number
        )

    def _calibrate(self, elbow_angle: float) -> None:
        """自适应校准：根据前30帧数据自动计算个性化阈值

        参数:
            elbow_angle: 当前肘部角度
        """
        if elbow_angle > 130:
            self._up_elbow_samples.append(elbow_angle)

        if self.frame_number >= self._calibration_frames:
            self._calibrated = True

            if self._up_elbow_samples:
                up_elbow = np.mean(self._up_elbow_samples)
            else:
                up_elbow = 170

            self.up_threshold = max(145, up_elbow * 0.90)
            self.descend_threshold = up_elbow * 0.81
            self.bottom_threshold = max(80, up_elbow * 0.54)

            self.config["elbow_angle_up_min"] = int(self.up_threshold)
            self.config["elbow_angle_up_max"] = 180
            self.config["elbow_angle_bottom_min"] = max(55, int(up_elbow * 0.34))
            self.config["elbow_angle_bottom_max"] = int(up_elbow * 0.62)

            self.config["elbow_angle_transition_min"] = max(50, int(up_elbow * 0.31))
            self.config["elbow_angle_transition_max"] = int(up_elbow * 0.96)

    def _get_standard_range(self, joint: str):
        """获取指定关节在当前阶段的标准角度范围

        参数:
            joint: 关节名称 ("elbow" 或 "shoulder")

        返回:
            (最小角度, 最大角度) 元组
        """
        if joint == "elbow":
            if self.current_phase == Phase.UP:
                return (self.config["elbow_angle_up_min"],
                        self.config["elbow_angle_up_max"])
            elif self.current_phase == Phase.BOTTOM:
                return (self.config["elbow_angle_bottom_min"],
                        self.config["elbow_angle_bottom_max"])
            else:
                return (self.config["elbow_angle_transition_min"],
                        self.config["elbow_angle_transition_max"])
        else:
            if self.current_phase == Phase.UP:
                return (self.config["shoulder_angle_up_min"],
                        self.config["shoulder_angle_up_max"])
            elif self.current_phase == Phase.BOTTOM:
                return (self.config["shoulder_angle_bottom_min"],
                        self.config["shoulder_angle_bottom_max"])
            else:
                return (self.config["shoulder_angle_transition_min"],
                        self.config["shoulder_angle_transition_max"])

    def _phase_name(self) -> str:
        """获取当前阶段的中文名称"""
        names = {Phase.UP: "撑起", Phase.DESCENDING: "下降",
                 Phase.BOTTOM: "底部", Phase.ASCENDING: "推起"}
        return names.get(self.current_phase, "")

    def _update_phase(self, elbow_angle: float) -> None:
        """更新动作阶段状态机

        参数:
            elbow_angle: 当前肘部角度
        """
        angle_diff = elbow_angle - self.previous_elbow_angle
        target_phase = None

        if self.current_phase == Phase.UP:
            if elbow_angle < self.descend_threshold and angle_diff < -1:
                target_phase = Phase.DESCENDING

        elif self.current_phase == Phase.DESCENDING:
            if elbow_angle < self.bottom_threshold and angle_diff < 1:
                target_phase = Phase.BOTTOM
            elif elbow_angle > self.up_threshold:
                target_phase = Phase.UP

        elif self.current_phase == Phase.BOTTOM:
            if angle_diff > 2:
                target_phase = Phase.ASCENDING

        elif self.current_phase == Phase.ASCENDING:
            if elbow_angle > self.up_threshold and angle_diff > -1:
                target_phase = Phase.UP
                if self._hit_bottom:
                    rep_duration = self.frame_number - self._rep_start_frame
                    if rep_duration >= self._min_rep_frames:
                        self.total_count += 1
                        if self.total_frames > 0 and self.standard_frames / max(self.total_frames, 1) > 0.5:
                            self.standard_count += 1
                self._hit_bottom = False
                self._rep_start_frame = 0

        if target_phase is not None:
            if target_phase == self._pending_phase:
                self._pending_count += 1
                if self._pending_count >= self.phase_debounce:
                    self.current_phase = target_phase
                    if target_phase == Phase.DESCENDING:
                        self.total_frames = 0
                        self.standard_frames = 0
                        self._rep_start_frame = self.frame_number
                        self._hit_bottom = False
                    elif target_phase == Phase.BOTTOM:
                        self._hit_bottom = True
                    self._pending_phase = None
                    self._pending_count = 0
            else:
                self._pending_phase = target_phase
                self._pending_count = 1
        else:
            self._pending_phase = None
            self._pending_count = 0

        self.previous_elbow_angle = elbow_angle
