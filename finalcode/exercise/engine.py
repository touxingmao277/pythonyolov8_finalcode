from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np
from ..pose_detector import PoseDetector, PoseResult
from ..angle_utils import AngleCalculator

@dataclass
class AngleReading:
    """关节角度的检测结果

    属性:
        name: 关节名称
        value: 检测到的角度值
        is_standard: 是否符合标准姿势
        min_standard: 标准角度范围最小值
        max_standard: 标准角度范围最大值
        joint_pos: 关节在图像中的位置坐标
    """
    name: str
    value: float
    is_standard: bool
    min_standard: float
    max_standard: float
    joint_pos: tuple = None

@dataclass
class Feedback:
    """动作分析反馈信息

    属性:
        is_standard: 当前姿势是否标准
        message: 状态消息
        issues: 不标准的问题列表
        suggestions: 改进建议列表
    """
    is_standard: bool
    message: str
    issues: List[str]
    suggestions: List[str]

@dataclass
class AnalysisResult:
    """动作分析结果

    属性:
        is_detected: 是否检测到人体
        angles: 各关节角度列表
        feedback: 动作反馈信息
        timestamp: 推理耗时
        frame_number: 帧编号
    """
    is_detected: bool
    angles: List[AngleReading]
    feedback: Feedback
    timestamp: float
    frame_number: int

@dataclass
class ExerciseStatistics:
    """运动统计信息

    属性:
        total_count: 总动作次数
        standard_count: 标准动作次数
        standard_rate: 标准率 (百分比)
        total_frames: 总帧数
        standard_frames: 标准帧数
    """
    total_count: int
    standard_count: int
    standard_rate: float
    total_frames: int
    standard_frames: int

class ExerciseAnalyzer(ABC):
    """动作分析器

    定义了动作分析器（俯卧撑、深蹲）需要实现的接口。
    支持自适应校准、阶段状态机、角度平滑和去抖动处理。
    """

    def __init__(self, pose_detector: PoseDetector,
                 angle_calculator: AngleCalculator) -> None:
        """初始化动作分析器

        参数:
            pose_detector: 姿态检测器实例
            angle_calculator: 角度计算器实例
        """
        self.pose_detector = pose_detector
        self.angle_calculator = angle_calculator
        self._reset_state()

    @abstractmethod
    def analyze(self, frame: np.ndarray, results: PoseResult) -> AnalysisResult:
        """分析单帧图像中的动作

        参数:
            frame: 输入图像
            results: 姿态检测结果

        返回:
            AnalysisResult 包含角度、反馈等信息
        """
        pass

    @abstractmethod
    def get_exercise_name(self) -> str:
        """获取动作名称

        返回:
            动作名称 ( "俯卧撑"、"深蹲")
        """
        pass

    def _reset_state(self) -> None:
        """重置分析器状态，开始新的分析"""
        self.total_count = 0
        self.standard_count = 0
        self.total_frames = 0
        self.standard_frames = 0

    def get_statistics(self) -> ExerciseStatistics:
        """获取当前运动统计信息

        返回:
            ExerciseStatistics 包含各项统计数据
        """
        standard_rate = (self.standard_count / max(self.total_count, 1)) * 100
        return ExerciseStatistics(
            total_count=self.total_count,
            standard_count=self.standard_count,
            standard_rate=standard_rate,
            total_frames=self.total_frames,
            standard_frames=self.standard_frames
        )