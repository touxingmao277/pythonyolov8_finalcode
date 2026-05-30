from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np
from ..pose_detector import PoseDetector, PoseResult
from ..angle_utils import AngleCalculator

@dataclass
class AngleReading:
    name: str
    value: float
    is_standard: bool
    min_standard: float
    max_standard: float
    joint_pos: tuple = None

@dataclass
class Feedback:
    is_standard: bool
    message: str
    issues: List[str]
    suggestions: List[str]

@dataclass
class AnalysisResult:
    is_detected: bool
    angles: List[AngleReading]
    feedback: Feedback
    timestamp: float
    frame_number: int

@dataclass
class ExerciseStatistics:
    total_count: int
    standard_count: int
    standard_rate: float
    total_frames: int
    standard_frames: int

class ExerciseAnalyzer(ABC):
    def __init__(self, pose_detector: PoseDetector,
                 angle_calculator: AngleCalculator) -> None:
        self.pose_detector = pose_detector
        self.angle_calculator = angle_calculator
        self._reset_state()

    @abstractmethod
    def analyze(self, frame: np.ndarray, results: PoseResult) -> AnalysisResult:
        pass

    @abstractmethod
    def get_exercise_name(self) -> str:
        pass

    def _reset_state(self) -> None:
        self.total_count = 0
        self.standard_count = 0
        self.total_frames = 0
        self.standard_frames = 0

    def get_statistics(self) -> ExerciseStatistics:
        standard_rate = (self.standard_count / max(self.total_count, 1)) * 100
        return ExerciseStatistics(
            total_count=self.total_count,
            standard_count=self.standard_count,
            standard_rate=standard_rate,
            total_frames=self.total_frames,
            standard_frames=self.standard_frames
        )