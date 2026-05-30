import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 关闭 ultralytics 逐帧推理日志
logging.getLogger("ultralytics").setLevel(logging.WARNING)

@dataclass
class Keypoint:
    id: int
    name: str
    x: float
    y: float
    confidence: float

@dataclass
class PoseResult:
    keypoints: List[Keypoint]
    box: Tuple[int, int, int, int]
    confidence: float
    timestamp: float

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# 左右对称关键点对，用于插值回退
LEFT_RIGHT_PAIRS = {
    5: 6, 6: 5,     # shoulders
    7: 8, 8: 7,     # elbows
    9: 10, 10: 9,   # wrists
    11: 12, 12: 11, # hips
    13: 14, 14: 13, # knees
    15: 16, 16: 15, # ankles
}

class PoseDetector:
    def __init__(
        self,
        model_name: str = "yolov8s-pose.pt",
        conf_threshold: float = 0.25,
        device: str = "auto"
    ) -> None:
        # 自动检测设备：无 CUDA 时回退到 CPU
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = device

        # 关键点历史缓存，用于短暂丢失时的插值
        self._kp_history: Dict[int, deque] = {}
        self._max_history = 3  # 最多回退 3 帧

    def detect_pose(self, frame: np.ndarray) -> Optional[PoseResult]:
        results = self.model(frame, conf=self.conf_threshold, device=self.device,
                            verbose=False)

        if not results or len(results[0].keypoints) == 0:
            return None

        keypoints_data = results[0].keypoints
        box_data = results[0].boxes

        if len(keypoints_data) == 0 or len(box_data) == 0:
            return None

        keypoints = []
        if hasattr(keypoints_data, 'xy') and keypoints_data.xy is not None:
            keypoints_xy = keypoints_data.xy.cpu().numpy() if hasattr(keypoints_data.xy, 'cpu') else keypoints_data.xy
            keypoints_conf = keypoints_data.conf.cpu().numpy() if hasattr(keypoints_data.conf, 'cpu') else keypoints_data.conf

            for i in range(len(KEYPOINT_NAMES)):
                if i < keypoints_xy.shape[1]:
                    x, y = keypoints_xy[0][i]
                    conf = float(keypoints_conf[0][i]) if keypoints_conf is not None and i < keypoints_conf.shape[1] else 0.0

                    # 更新历史缓存
                    if conf >= self.conf_threshold:
                        if i not in self._kp_history:
                            self._kp_history[i] = deque(maxlen=self._max_history)
                        self._kp_history[i].append((float(x), float(y)))

                    keypoints.append(Keypoint(
                        id=i, name=KEYPOINT_NAMES[i],
                        x=float(x), y=float(y), confidence=conf
                    ))
                else:
                    keypoints.append(Keypoint(
                        id=i, name=KEYPOINT_NAMES[i],
                        x=0.0, y=0.0, confidence=0.0
                    ))

        box = box_data[0].xyxy[0]
        timestamp = results[0].speed.get('inference', 0.0)

        return PoseResult(
            keypoints=keypoints,
            box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            confidence=float(box_data[0].conf[0]),
            timestamp=timestamp
        )

    def get_keypoint_coords(self, keypoint_id: int, results: PoseResult) -> Optional[Tuple[float, float]]:
        if keypoint_id < 0 or keypoint_id >= len(results.keypoints):
            return None
        keypoint = results.keypoints[keypoint_id]
        if keypoint.confidence >= self.conf_threshold:
            return (keypoint.x, keypoint.y)

        # 置信度不足：尝试从历史缓存插值
        if keypoint_id in self._kp_history and len(self._kp_history[keypoint_id]) > 0:
            return self._kp_history[keypoint_id][-1]

        return None

    def get_keypoint_with_fallback(self, primary_id: int, fallback_id: int,
                                    results: PoseResult) -> Optional[Tuple[float, float]]:
        """获取关键点坐标，优先主侧，失败时自动回退到对侧，再失败则用历史"""
        # 尝试主侧
        coords = self.get_keypoint_coords(primary_id, results)
        if coords is not None:
            return coords
        # 尝试对侧
        coords = self.get_keypoint_coords(fallback_id, results)
        if coords is not None:
            return coords
        # 最后尝试历史
        for kp_id in (primary_id, fallback_id):
            if kp_id in self._kp_history and len(self._kp_history[kp_id]) > 0:
                return self._kp_history[kp_id][-1]
        return None

    def get_best_keypoint(self, left_id: int, right_id: int,
                          results: PoseResult) -> Optional[Tuple[float, float]]:
        """获取左右两侧中置信度更高的关键点"""
        left_kp = self.get_keypoint_coords(left_id, results)
        right_kp = self.get_keypoint_coords(right_id, results)
        if left_kp is not None and right_kp is not None:
            left_conf = results.keypoints[left_id].confidence
            right_conf = results.keypoints[right_id].confidence
            return left_kp if left_conf >= right_conf else right_kp
        return left_kp if left_kp is not None else right_kp

    def draw_skeleton(self, frame: np.ndarray, results: PoseResult,
                      line_color: Tuple[int, int, int] = (0, 255, 0),
                      point_color: Tuple[int, int, int] = (0, 255, 0),
                      point_radius: int = 3,
                      line_thickness: int = 2) -> np.ndarray:
        frame_copy = frame.copy()

        for connection in SKELETON_CONNECTIONS:
            point1 = self.get_keypoint_coords(connection[0], results)
            point2 = self.get_keypoint_coords(connection[1], results)

            if point1 is not None and point2 is not None:
                cv2.line(frame_copy, (int(point1[0]), int(point1[1])),
                         (int(point2[0]), int(point2[1])), line_color, line_thickness)

        for keypoint in results.keypoints:
            if keypoint.confidence >= self.conf_threshold:
                cv2.circle(frame_copy, (int(keypoint.x), int(keypoint.y)),
                           point_radius, point_color, -1)

        return frame_copy
