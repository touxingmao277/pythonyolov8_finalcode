from collections import deque
from enum import Enum
import numpy as np
from ..pose_detector import PoseDetector, PoseResult
from ..angle_utils import AngleCalculator
from .engine import ExerciseAnalyzer, AngleReading, Feedback, AnalysisResult

class Phase(Enum):
    UP = 0
    DESCENDING = 1
    BOTTOM = 2
    ASCENDING = 3

class PushUpAnalyzer(ExerciseAnalyzer):
    def __init__(self, pose_detector: PoseDetector,
                 angle_calculator: AngleCalculator) -> None:
        super().__init__(pose_detector, angle_calculator)

        # ---- 硬编码默认值（校准前使用） ----
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

        # ---- 自适应校准 ----
        self._calibrated = False
        self._calibration_frames = 30
        self._up_elbow_samples = []

        self.descend_threshold = 140
        self.bottom_threshold = 95
        self.up_threshold = 155

        # ---- 角度平滑（3 帧移动平均） ----
        self._elbow_buffer = deque(maxlen=3)
        self._shoulder_buffer = deque(maxlen=3)

        # ---- 时序去抖动 ----
        self.phase_debounce = 5
        self._pending_phase = None
        self._pending_count = 0

        # ---- 防虚高 ----
        self._hit_bottom = False
        self._rep_start_frame = 0
        self._min_rep_frames = 15

    def get_exercise_name(self) -> str:
        return "俯卧撑"

    def analyze(self, frame: np.ndarray, results: PoseResult) -> AnalysisResult:
        self.frame_number += 1

        if results is None or len(results.keypoints) < 3:
            return AnalysisResult(
                is_detected=False, angles=[],
                feedback=Feedback(is_standard=False, message="未检测到人体",
                                  issues=[], suggestions=[]),
                timestamp=0.0, frame_number=self.frame_number
            )

        # 关键点获取：左侧优先 → 右侧回退 → 历史插值
        shoulder = self.pose_detector.get_keypoint_with_fallback(5, 6, results)
        elbow = self.pose_detector.get_keypoint_with_fallback(7, 8, results)
        wrist = self.pose_detector.get_keypoint_with_fallback(9, 10, results)
        hip = self.pose_detector.get_keypoint_with_fallback(11, 12, results)
        ankle = self.pose_detector.get_keypoint_with_fallback(15, 16, results)

        angles = []
        issues = []
        suggestions = []
        elbow_angle = None
        body_angle = None  # 身体直线度

        # === 肘部角度（肩-肘-腕） ===
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

        # === 身体直线度（肩-髋-踝） ===
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

        # === 自适应校准 ===
        if not self._calibrated and elbow_angle is not None:
            self._calibrate(elbow_angle)

        # === 阶段状态机 ===
        if elbow_angle is not None:
            self._update_phase(elbow_angle)

        # === 判断标准 ===
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
        """前 N 帧收集撑起姿态数据，计算个性化阈值"""
        if elbow_angle > 130:
            self._up_elbow_samples.append(elbow_angle)

        if self.frame_number >= self._calibration_frames:
            self._calibrated = True

            if self._up_elbow_samples:
                up_elbow = np.mean(self._up_elbow_samples)
            else:
                up_elbow = 170

            # 从撑起角度推导阈值
            self.up_threshold = max(145, up_elbow * 0.90)
            self.descend_threshold = up_elbow * 0.81
            self.bottom_threshold = max(80, up_elbow * 0.54)

            # 更新标准范围
            self.config["elbow_angle_up_min"] = int(self.up_threshold)
            self.config["elbow_angle_up_max"] = 180
            self.config["elbow_angle_bottom_min"] = max(55, int(up_elbow * 0.34))
            self.config["elbow_angle_bottom_max"] = int(up_elbow * 0.62)

            self.config["elbow_angle_transition_min"] = max(50, int(up_elbow * 0.31))
            self.config["elbow_angle_transition_max"] = int(up_elbow * 0.96)

    def _get_standard_range(self, joint: str):
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
        else:  # shoulder / body alignment
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
        names = {Phase.UP: "撑起", Phase.DESCENDING: "下降",
                 Phase.BOTTOM: "底部", Phase.ASCENDING: "推起"}
        return names.get(self.current_phase, "")

    def _update_phase(self, elbow_angle: float) -> None:
        angle_diff = elbow_angle - self.previous_elbow_angle
        target_phase = None

        if self.current_phase == Phase.UP:
            if elbow_angle < self.descend_threshold and angle_diff < -1:
                target_phase = Phase.DESCENDING

        elif self.current_phase == Phase.DESCENDING:
            if elbow_angle < self.bottom_threshold and angle_diff < 1:
                target_phase = Phase.BOTTOM
            elif elbow_angle > self.up_threshold:
                target_phase = Phase.UP  # 放弃

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

        # 去抖动
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
