import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.WARNING)

@dataclass
class Keypoint:
    """人体关键点数据类

    属性:
        id: 关键点编号 (0-16，对应COCO 17点姿态)
        name: 关键点名称 (对应KEYPOINT_NAMES列表中的名称)
        x: 关键点在图像中的横坐标 (0-1)
        y: 关键点在图像中的纵坐标
        confidence: 关键点检测置信度 (0-1) 越高表示检测越准确但识别识别不出来
    """
    id: int
    name: str
    x: float
    y: float
    confidence: float

@dataclass
class PoseResult:
    """姿态检测结果数据类

    属性:
        keypoints: 检测到的关键点列表
        box: 人物边界框坐标 (x1, y1, x2, y2)
        confidence: 整体检测置信度
        timestamp: 推理耗时 (ms)
    """
    keypoints: List[Keypoint]
    box: Tuple[int, int, int, int]
    confidence: float
    timestamp: float

# 骨架连接关系 (对应KEYPOINT_NAMES列表中的索引)
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

# 关键点名称列表
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# 关键点左右对应关系 (双向映射)
LEFT_RIGHT_PAIRS = {
    5: 6, 6: 5,
    7: 8, 8: 7,
    9: 10, 10: 9,
    11: 12, 12: 11,
    13: 14, 14: 13,
    15: 16, 16: 15,
}

class PoseDetector:
    def __init__(
        self,
        model_name: str = "yolov8s-pose.pt",
        conf_threshold: float = 0.25,
        device: str = "auto"
    ) -> None:
        """初始化姿态检测器

        参数:
            model_name: YOLOv8模型文件名
            conf_threshold: 低于此值的关键点将被忽略
            device: 运行设备，"auto"自动选择GPU/CPU，"cuda"强制使用GPU，"cpu"强制使用CPU
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = device

        self._kp_history: Dict[int, deque] = {} # 关键点位置历史记录 deque是一个双端队列
        self._max_history = 3 # 最大历史记录数，用于平滑关键点位置

    def detect_pose(self, frame: np.ndarray) -> Optional[PoseResult]:
        """检测单帧图像中的人体姿态

        参数:
            frame: 输入图像 (BGR格式，numpy数组)

        返回:
            PoseResult对象包含检测到的关键点、边界框等信息
            如果未检测到人体则返回None (Optional)
        """
        results = self.model(frame, conf=self.conf_threshold, device=self.device,
                            verbose=False) # verbose否输出详细信息

        # 检查检测结果是否为空
        if not results or len(results[0].keypoints) == 0:
            return None

        # 提取关键点数据
        keypoints_data = results[0].keypoints
        box_data = results[0].boxes

        # 检查关键点数据是否为空
        if len(keypoints_data) == 0 or len(box_data) == 0:
            return None

        keypoints = []
        # hasattr (内置函数)检查关键点数据是否包含xy属性
        if hasattr(keypoints_data, 'xy') and keypoints_data.xy is not None:
            # 提取关键点坐标和置信度
            # keypoints_xy 通常的形状是 (人数, 关键点数, 坐标维度)
            keypoints_xy = keypoints_data.xy.cpu().numpy() if hasattr(keypoints_data.xy, 'cpu') else keypoints_data.xy
            keypoints_conf = keypoints_data.conf.cpu().numpy() if hasattr(keypoints_data.conf, 'cpu') else keypoints_data.conf

            # 遍历关键点，提取坐标和置信度
            for i in range(len(KEYPOINT_NAMES)):
                if i < keypoints_xy.shape[1]: # 检查关键点索引是否有效
                    x, y = keypoints_xy[0][i]
                    conf = float(keypoints_conf[0][i]) if keypoints_conf is not None and i < keypoints_conf.shape[1] else 0.0

                    # 筛选出置信度大于阈值的关键点
                    if conf >= self.conf_threshold:
                        # 更新关键点位置历史记录
                        if i not in self._kp_history:
                            # 初始化关键点位置历史记录
                            self._kp_history[i] = deque(maxlen=self._max_history)# deque自动维护最大长度
                        self._kp_history[i].append((float(x), float(y)))

                    # 添加有效关键点到结果
                    keypoints.append(Keypoint(
                        id=i, name=KEYPOINT_NAMES[i],
                        x=float(x), y=float(y), confidence=conf
                    ))
                else:
                    keypoints.append(Keypoint(
                        id=i, name=KEYPOINT_NAMES[i],
                        x=0.0, y=0.0, confidence=0.0
                    ))

        # 提取边界框数据
        box = box_data[0].xyxy[0]
        # 提取推理时间戳
        timestamp = results[0].speed.get('inference', 0.0)

        return PoseResult(
            keypoints=keypoints,
            box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            confidence=float(box_data[0].conf[0]),
            timestamp=timestamp
        )

    def get_keypoint_coords(self, keypoint_id: int, results: PoseResult) -> Optional[Tuple[float, float]]:
        """获取指定关键点的坐标

        参数:
            keypoint_id: 关键点编号 (0-16)
            results: 姿态检测结果

        返回:
            关键点坐标 (x, y)，如果置信度不足且无历史数据则返回None
        """
        # 检查关键点ID是否有效
        if keypoint_id < 0 or keypoint_id >= len(results.keypoints):
            return None
        keypoint = results.keypoints[keypoint_id]
        # 检查关键点置信度是否有效
        if keypoint.confidence >= self.conf_threshold:
            return (keypoint.x, keypoint.y)

        # 检查关键点是否有历史记录
        if keypoint_id in self._kp_history and len(self._kp_history[keypoint_id]) > 0:
            # 返回最近一次记录的坐标
            return self._kp_history[keypoint_id][-1]

        return None

    def get_keypoint_with_fallback(self, primary_id: int, fallback_id: int,
                                    results: PoseResult) -> Optional[Tuple[float, float]]:
        """获取关键点坐标，支持左右对称回退

        优先返回主侧关键点，如果主侧置信度不足则尝试对侧，
        最后回退到历史缓存数据。

        参数:
            primary_id: 主侧关键点编号 (如左肩 5)
            fallback_id: 回退侧关键点编号 (如右肩 6)
            results: 姿态检测结果

        返回:
            可用的关键点坐标或None
        """
        coords = self.get_keypoint_coords(primary_id, results)
        if coords is not None:
            return coords
        coords = self.get_keypoint_coords(fallback_id, results)
        if coords is not None:
            return coords
        for kp_id in (primary_id, fallback_id):
            if kp_id in self._kp_history and len(self._kp_history[kp_id]) > 0:
                return self._kp_history[kp_id][-1]
        return None

    def get_best_keypoint(self, left_id: int, right_id: int,
                          results: PoseResult) -> Optional[Tuple[float, float]]:
        """获取左右两侧中置信度更高的关键点

        参数:
            left_id: 左侧关键点编号
            right_id: 右侧关键点编号
            results: 姿态检测结果

        返回:
            置信度更高的关键点坐标
        """
        # 获取左右两侧关键点坐标
        left_kp = self.get_keypoint_coords(left_id, results)
        right_kp = self.get_keypoint_coords(right_id, results)
        # 检查左右两侧关键点是否有坐标
        if left_kp is not None and right_kp is not None:
            left_conf = results.keypoints[left_id].confidence
            right_conf = results.keypoints[right_id].confidence
            # 比较左右两侧关键点置信度，返回置信度更高的坐标
            return left_kp if left_conf >= right_conf else right_kp
        # 返回第一个非None的坐标
        return left_kp if left_kp is not None else right_kp

    def draw_skeleton(self, frame: np.ndarray, results: PoseResult,
                      line_color: Tuple[int, int, int] = (0, 255, 0),
                      point_color: Tuple[int, int, int] = (0, 255, 0),
                      point_radius: int = 3,
                      line_thickness: int = 2) -> np.ndarray: # ndarray用于存储图像数据
        """在图像上绘制人体骨骼线

        参数:
            frame: 输入图像
            results: 姿态检测结果
            line_color: 骨骼线颜色 (BGR格式)
            point_color: 关键点颜色 (BGR格式)
            point_radius: 关键点圆圈半径
            line_thickness: 骨骼线粗细

        返回:
            绘制了骨骼线的图像副本
        """
        # 复制输入图像
        frame_copy = frame.copy()

        # 遍历骨骼连接
        for connection in SKELETON_CONNECTIONS:
            # 获取骨骼连接的两个关键点坐标
            point1 = self.get_keypoint_coords(connection[0], results)
            point2 = self.get_keypoint_coords(connection[1], results)
            # 检查关键点是否有坐标
            if point1 is not None and point2 is not None:
                # 绘制骨骼线
                cv2.line(frame_copy, (int(point1[0]), int(point1[1])),
                         (int(point2[0]), int(point2[1])), line_color, line_thickness)

        # 遍历所有关键点
        for keypoint in results.keypoints:
            # 检查关键点置信度是否有效
            if keypoint.confidence < self.conf_threshold:
                continue
            # 绘制关键点
            if keypoint.confidence >= self.conf_threshold:
                # 绘制关键点
                cv2.circle(frame_copy, (int(keypoint.x), int(keypoint.y)),
                           point_radius, point_color, -1)

        return frame_copy
