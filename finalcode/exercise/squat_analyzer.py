from collections import deque
from enum import Enum
import numpy as np
from ..pose_detector import PoseDetector, PoseResult
from ..angle_utils import AngleCalculator
from .engine import ExerciseAnalyzer, AngleReading, Feedback, AnalysisResult

class Phase(Enum):
    """深蹲动作阶段枚举"""
    STANDING = 0    # 站立状态
    DESCENDING = 1  # 下蹲中
    BOTTOM = 2      # 底部
    ASCENDING = 3   # 起身中

class SquatAnalyzer(ExerciseAnalyzer):
    """深蹲动作分析器

    通过分析人体关键点角度变化，检测深蹲动作的完成情况。
    支持自适应校准（根据用户站立姿势调整判定阈值）、
    三帧移动平均平滑处理、以及阶段状态机逻辑。
    分析的关键角度包括：
    - 膝盖角度（髋-膝-踝）
    - 髋部角度（肩-髋-膝），反映躯干前倾程度
    """

    def __init__(self, pose_detector: PoseDetector,
                 angle_calculator: AngleCalculator) -> None:
        """初始化深蹲分析器

        参数:
            pose_detector: 姿态检测器实例
            angle_calculator: 角度计算器实例
        """
        super().__init__(pose_detector, angle_calculator)

        self.config = {
            "knee_angle_bottom_min": 70,
            "knee_angle_bottom_max": 110,
            "hip_angle_bottom_min": 40,
            "hip_angle_bottom_max": 80,
            "knee_angle_stand_min": 155,
            "knee_angle_stand_max": 180,
            "hip_angle_stand_min": 155,
            "hip_angle_stand_max": 180,
            "knee_angle_transition_min": 85,
            "knee_angle_transition_max": 160,
            "hip_angle_transition_min": 35,
            "hip_angle_transition_max": 165,
        }

        self.current_phase = Phase.STANDING
        self.previous_knee_angle = 180.0
        self.frame_number = 0

        self._calibrated = False
        self._calibration_frames = 30
        self._standing_knee_samples = []
        self._standing_hip_samples = []

        self.descend_threshold = 150
        self.bottom_threshold = 90
        self.stand_threshold = 160

        self._knee_buffer = deque(maxlen=3)
        self._hip_buffer = deque(maxlen=3)

        self.phase_debounce = 5
        self._pending_phase = None
        self._pending_count = 0

        self._hit_bottom = False
        self._rep_start_frame = 0
        self._min_rep_frames = 15

    def get_exercise_name(self) -> str:
        return "深蹲"

    def analyze(self, frame: np.ndarray, results: PoseResult) -> AnalysisResult:
        """分析单帧图像中的深蹲动作

        参数:
            frame: 输入图像
            results: 姿态检测结果

        返回:
            AnalysisResult 包含膝盖角度、髋部角度、阶段状态和反馈信息
        """
        self.frame_number += 1

        if results is None or len(results.keypoints) < 3:
            return AnalysisResult(
                is_detected=False, angles=[],
                feedback=Feedback(is_standard=False, message="未检测到人体",
                                  issues=[], suggestions=[]),
                timestamp=0.0, frame_number=self.frame_number
            )

        hip = self.pose_detector.get_keypoint_with_fallback(11, 12, results)
        knee = self.pose_detector.get_keypoint_with_fallback(13, 14, results)
        ankle = self.pose_detector.get_keypoint_with_fallback(15, 16, results)
        shoulder = self.pose_detector.get_keypoint_with_fallback(5, 6, results)

        angles = []
        issues = []
        suggestions = []
        knee_angle = None
        hip_angle = None

        if hip is not None and knee is not None and ankle is not None:
            raw = self.angle_calculator.calculate_angle(hip, knee, ankle)
            self._knee_buffer.append(raw)
            knee_angle = np.mean(self._knee_buffer) if self._knee_buffer else raw

            knee_min, knee_max = self._get_standard_range("knee")
            knee_is_standard = self.angle_calculator.is_angle_in_range(
                knee_angle, knee_min, knee_max)

            angles.append(AngleReading(
                "膝盖角度", knee_angle, knee_is_standard,
                knee_min, knee_max, joint_pos=knee))

            if not knee_is_standard:
                issues.append(f"膝盖: {knee_angle:.0f}° (标准 {knee_min}-{knee_max}°)")
                if self.current_phase == Phase.STANDING:
                    suggestions.append("站直，膝盖微屈")
                elif self.current_phase == Phase.BOTTOM:
                    suggestions.append("蹲至大腿与地面平行")

        if shoulder is not None and hip is not None and knee is not None:
            raw_hip = self.angle_calculator.calculate_angle(shoulder, hip, knee)
            self._hip_buffer.append(raw_hip)
            hip_angle = np.mean(self._hip_buffer) if self._hip_buffer else raw_hip

            hip_min, hip_max = self._get_standard_range("hip")
            hip_is_standard = self.angle_calculator.is_angle_in_range(
                hip_angle, hip_min, hip_max)

            angles.append(AngleReading(
                "髋部角度", hip_angle, hip_is_standard,
                hip_min, hip_max, joint_pos=hip))

            if not hip_is_standard:
                issues.append(f"髋部: {hip_angle:.0f}° (标准 {hip_min}-{hip_max}°)")
                suggestions.append("挺直躯干，不要过度前倾")

        if not angles:
            return AnalysisResult(
                is_detected=True, angles=[],
                feedback=Feedback(is_standard=False, message="关键点不足",
                                  issues=[], suggestions=[]),
                timestamp=results.timestamp if results else 0.0,
                frame_number=self.frame_number
            )

        if not self._calibrated and knee_angle is not None:
            self._calibrate(knee_angle, hip_angle)

        if knee_angle is not None:
            self._update_phase(knee_angle)

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

    def _calibrate(self, knee_angle: float, hip_angle: float) -> None:
        """自适应校准：根据前30帧站立数据自动计算个性化阈值

        参数:
            knee_angle: 当前膝盖角度
            hip_angle: 当前髋部角度
        """
        if knee_angle > 140:
            self._standing_knee_samples.append(knee_angle)
        if hip_angle is not None and hip_angle > 140:
            self._standing_hip_samples.append(hip_angle)

        if self.frame_number >= self._calibration_frames:
            self._calibrated = True

            if self._standing_knee_samples:
                stand_knee = np.mean(self._standing_knee_samples)
            else:
                stand_knee = 170

            if self._standing_hip_samples:
                stand_hip = np.mean(self._standing_hip_samples)
            else:
                stand_hip = 170

            self.stand_threshold = max(150, stand_knee * 0.93)
            self.descend_threshold = stand_knee * 0.87
            self.bottom_threshold = max(75, stand_knee * 0.53)

            self.config["knee_angle_stand_min"] = int(self.stand_threshold)
            self.config["knee_angle_stand_max"] = 180
            self.config["knee_angle_bottom_min"] = max(65, int(stand_knee * 0.40))
            self.config["knee_angle_bottom_max"] = int(stand_knee * 0.65)

            self.config["hip_angle_stand_min"] = int(stand_hip * 0.90)
            self.config["hip_angle_stand_max"] = 180
            self.config["hip_angle_bottom_min"] = max(35, int(stand_hip * 0.25))
            self.config["hip_angle_bottom_max"] = int(stand_hip * 0.50)

            self.config["knee_angle_transition_min"] = max(75, int(stand_knee * 0.48))
            self.config["knee_angle_transition_max"] = int(stand_knee * 0.95)
            self.config["hip_angle_transition_min"] = max(35, int(stand_hip * 0.22))
            self.config["hip_angle_transition_max"] = int(stand_hip * 0.95)

    def _get_standard_range(self, joint: str):
        """获取指定关节在当前阶段的标准角度范围

        参数:
            joint: 关节名称 ("knee" 或 "hip")

        返回:
            (最小角度, 最大角度) 元组
        """
        if joint == "knee":
            if self.current_phase == Phase.STANDING:
                return (self.config["knee_angle_stand_min"],
                        self.config["knee_angle_stand_max"])
            elif self.current_phase == Phase.BOTTOM:
                return (self.config["knee_angle_bottom_min"],
                        self.config["knee_angle_bottom_max"])
            else:
                return (self.config["knee_angle_transition_min"],
                        self.config["knee_angle_transition_max"])
        else:
            if self.current_phase == Phase.STANDING:
                return (self.config["hip_angle_stand_min"],
                        self.config["hip_angle_stand_max"])
            elif self.current_phase == Phase.BOTTOM:
                return (self.config["hip_angle_bottom_min"],
                        self.config["hip_angle_bottom_max"])
            else:
                return (self.config["hip_angle_transition_min"],
                        self.config["hip_angle_transition_max"])

    def _phase_name(self) -> str:
        """获取当前阶段的中文名称"""
        names = {Phase.STANDING: "站立", Phase.DESCENDING: "下蹲",
                 Phase.BOTTOM: "底部", Phase.ASCENDING: "起身"}
        return names.get(self.current_phase, "")

    def _update_phase(self, knee_angle: float) -> None:
        """更新动作阶段状态机

        参数:
            knee_angle: 当前膝盖角度
        """
        angle_diff = knee_angle - self.previous_knee_angle
        target_phase = None

        if self.current_phase == Phase.STANDING:
            if knee_angle < self.descend_threshold and angle_diff < -1:
                target_phase = Phase.DESCENDING

        elif self.current_phase == Phase.DESCENDING:
            if knee_angle < self.bottom_threshold and angle_diff < 1:
                target_phase = Phase.BOTTOM
            elif knee_angle > self.stand_threshold:
                target_phase = Phase.STANDING

        elif self.current_phase == Phase.BOTTOM:
            if angle_diff > 2:
                target_phase = Phase.ASCENDING

        elif self.current_phase == Phase.ASCENDING:
            if knee_angle > self.stand_threshold and angle_diff > -1:
                target_phase = Phase.STANDING
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

        self.previous_knee_angle = knee_angle
