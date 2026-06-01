# 人体动作分析系统 - Vibe Coding 实现 Prompt

## 项目概述

### 项目名称
人体动作分析系统（Human Pose Analysis System）

### 项目目标
开发一套基于计算机视觉的人体动作分析系统，利用 YOLOv8-pose 姿态检测技术，实现对**深蹲**和**俯卧撑**两种常见运动的实时检测与标准性评估，为用户提供客观、即时的动作反馈。

### 技术栈
- **编程语言**：Python 3.8+
- **深度学习框架**：PyTorch 1.8+
- **姿态检测模型**：YOLOv8-pose 8.0+ (ultralytics)
- **计算机视觉库**：OpenCV 4.5+
- **GUI框架**：待定（可选 PyQt5/Tkinter）

### 项目目录结构

```
python大作业/
├── finalcode/
│   ├── __init__.py
│   ├── pose_detector.py          # 姿态检测器
│   ├── angle_utils.py            # 角度计算工具
│   ├── exercise/
│   │   ├── __init__.py
│   │   ├── engine.py              # 运动引擎抽象基类
│   │   ├── squat_analyzer.py      # 深蹲分析器
│   │   └── pushup_analyzer.py     # 俯卧撑分析器
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── video_analyzer.py      # 视频文件分析器
│   │   └── camera_analyzer.py      # 摄像头实时分析器
│   ├── cli.py                    # 命令行界面
│   └── gui.py                    # 图形用户界面
├── tests/
│   ├── __init__.py
│   ├── test_pose_detector.py
│   ├── test_angle_calculator.py
│   ├── test_squat_analyzer.py
│   ├── test_pushup_analyzer.py
│   ├── test_video_analyzer.py
│   └── test_camera_analyzer.py
├── doc/
│   ├── proposal.md               # 需求文档
│   ├── detailed-design.md         # 详细设计文档
│   └── tasks/                     # 任务划分
└── requirements.txt
```

---

## 核心模块实现要求

### 模块一：PoseDetector（姿态检测器）

**文件路径**：`finalcode/pose_detector.py`

**功能描述**：基于 YOLOv8-pose 的人体关键点检测与骨骼绘制

**类定义**：
```python
class PoseDetector:
    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        conf_threshold: float = 0.5,
        device: str = "auto"
    ) -> None

    def detect_pose(self, frame: np.ndarray) -> PoseResult
    def get_keypoint_coords(self, keypoint_id: int, results: PoseResult) -> Optional[Tuple[float, float]]
    def draw_skeleton(self, frame: np.ndarray, results: PoseResult,
                      line_color: Tuple[int, int, int] = (0, 255, 0),
                      point_color: Tuple[int, int, int] = (0, 255, 0),
                      point_radius: int = 3,
                      line_thickness: int = 2) -> np.ndarray
```

**数据结构**：
```python
@dataclass
class Keypoint:
    id: int           # COCO格式关键点索引 (0-16)
    name: str         # 关键点名称
    x: float          # 图像坐标 x
    y: float          # 图像坐标 y
    confidence: float # 置信度 (0-1)

@dataclass
class PoseResult:
    keypoints: List[Keypoint]  # 检测到的关键点列表
    box: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    confidence: float  # 整体置信度
    timestamp: float   # 检测时间戳
```

**COCO 关键点定义**：
| 索引 | 名称 | 描述 |
|------|------|------|
| 0 | nose | 鼻子 |
| 1 | left_eye | 左眼 |
| 2 | right_eye | 右眼 |
| 3 | left_ear | 左耳 |
| 4 | right_ear | 右耳 |
| 5 | left_shoulder | 左肩 |
| 6 | right_shoulder | 右肩 |
| 7 | left_elbow | 左肘 |
| 8 | right_elbow | 右肘 |
| 9 | left_wrist | 左腕 |
| 10 | right_wrist | 右腕 |
| 11 | left_hip | 左髋 |
| 12 | right_hip | 右髋 |
| 13 | left_knee | 左膝 |
| 14 | right_knee | 右膝 |
| 15 | left_ankle | 左踝 |
| 16 | right_ankle | 右踝 |

**骨骼连接定义**：
```python
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6),                           # 肩膀
    (5, 11), (6, 12),               # 肩膀到髋部
    (11, 12),                        # 髋部
    (5, 7), (7, 9),                 # 左臂
    (6, 8), (8, 10),                # 右臂
    (11, 13), (13, 15),             # 左腿
    (12, 14), (14, 16),             # 右腿
]
```

---

### 模块二：AngleCalculator（角度计算器）

**文件路径**：`finalcode/angle_utils.py`

**功能描述**：三维/二维向量角度计算工具

**类定义**：
```python
class AngleCalculator:
    @staticmethod
    def calculate_angle(a: Tuple[float, float],
                        b: Tuple[float, float],
                        c: Tuple[float, float]) -> float

    @staticmethod
    def calculate_angle_3d(a: Tuple[float, float, float],
                           b: Tuple[float, float, float],
                           c: Tuple[float, float, float]) -> float

    @staticmethod
    def is_angle_in_range(angle: float, min_angle: float,
                          max_angle: float, tolerance: float = 5.0) -> bool

    @staticmethod
    def normalize_angle(angle: float) -> float
```

**计算公式**：
```
向量 BA = (ax - bx, ay - by)
向量 BC = (cx - bx, cy - by)
cos(∠B) = (BA·BC) / (|BA|·|BC|)
∠B = arccos((BA·BC) / (|BA|·|BC|))
```

---

### 模块三：ExerciseEngine（运动引擎抽象基类）

**文件路径**：`finalcode/exercise/engine.py`

**功能描述**：动作分析引擎的抽象基类，定义分析器通用接口

**类定义**：
```python
from abc import ABC, abstractmethod

class ExerciseAnalyzer(ABC):
    def __init__(self, pose_detector: PoseDetector,
                 angle_calculator: AngleCalculator) -> None

    @abstractmethod
    def analyze(self, frame: np.ndarray, results: PoseResult) -> AnalysisResult

    @abstractmethod
    def get_exercise_name(self) -> str

    def _reset_state(self) -> None
    def get_statistics(self) -> ExerciseStatistics
```

**数据结构**：
```python
@dataclass
class AngleReading:
    name: str           # 角度名称
    value: float        # 角度值
    is_standard: bool   # 是否符合标准
    min_standard: float # 标准范围最小值
    max_standard: float # 标准范围最大值

@dataclass
class Feedback:
    is_standard: bool           # 整体是否标准
    message: str                # 反馈信息文本
    issues: List[str]           # 具体问题列表
    suggestions: List[str]      # 改进建议列表

@dataclass
class AnalysisResult:
    is_detected: bool                      # 是否检测到有效姿态
    angles: List[AngleReading]             # 各关节角度读数
    feedback: Feedback                      # 动作反馈
    timestamp: float                        # 时间戳
    frame_number: int                       # 帧编号

@dataclass
class ExerciseStatistics:
    total_count: int      # 总动作次数
    standard_count: int   # 标准动作次数
    standard_rate: float  # 标准率 (%)
    total_frames: int     # 总检测帧数
    standard_frames: int  # 标准帧数
```

---

### 模块四：SquatAnalyzer（深蹲分析器）

**文件路径**：`finalcode/exercise/squat_analyzer.py`

**功能描述**：深蹲动作的实时分析与标准评估

**阶段枚举**：
```python
class Phase(Enum):
    STANDING = 0      # 站立
    DESCENDING = 1    # 下降
    BOTTOM = 2        # 最低点
    ASCENDING = 3     # 上升
```

**配置参数**：
```python
self.config = {
    "knee_angle_min": 70,
    "knee_angle_max": 110,
    "hip_angle_min": 70,
    "hip_angle_max": 100,
    "depth_threshold": 0.9,
}
```

**角度计算**：
- 膝盖角度：left_hip(11) → left_knee(13) → left_ankle(15)
- 髋部角度：left_shoulder(5) → left_hip(11) → left_knee(13)

**状态机转换**：
```
STANDING → DESCENDING (knee_angle < 150°)
DESCENDING → BOTTOM (knee_angle < 90°)
BOTTOM → ASCENDING (knee_angle > 150°)
ASCENDING → STANDING (knee_angle > 160°) → 计数+1
```

---

### 模块五：PushUpAnalyzer（俯卧撑分析器）

**文件路径**：`finalcode/exercise/pushup_analyzer.py`

**功能描述**：俯卧撑动作的实时分析与标准评估

**阶段枚举**：与深蹲类似（STANDING, DESCENDING, BOTTOM, ASCENDING）

**配置参数**：
```python
self.config = {
    "elbow_angle_min": 60,
    "elbow_angle_max": 100,
    "shoulder_angle_min": 170,
    "shoulder_angle_max": 180,
}
```

**角度计算**：
- 肘部角度：left_shoulder(5) → left_elbow(7) → left_wrist(9)
- 肩部角度：left_hip(11) → left_shoulder(5) → left_elbow(7)

**评估标准**：
- 肘部角度范围：60° - 100°
- 肩部角度范围：170° - 180°
- 身体保持直线：头部、躯干、腿部成一条线
- 核心收紧
- 头部保持中立位置

---

### 模块六：VideoAnalyzer（视频分析器）

**文件路径**：`finalcode/analyzers/video_analyzer.py`

**功能描述**：视频文件离线分析

**类定义**：
```python
class VideoAnalyzer:
    def __init__(self, analyzer: ExerciseAnalyzer) -> None

    def analyze_video(self, video_path: str, output_path: str = None,
                     show_preview: bool = True) -> ExerciseStatistics

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult
```

**支持格式**：MP4、AVI、MOV、WebM

---

### 模块七：CameraAnalyzer（摄像头实时分析器）

**文件路径**：`finalcode/analyzers/camera_analyzer.py`

**功能描述**：摄像头实时分析

**类定义**：
```python
class CameraAnalyzer:
    def __init__(self, analyzer: ExerciseAnalyzer,
                 camera_id: int = 0, width: int = 640,
                 height: int = 480) -> None

    def start(self, window_name: str = "Pose Analysis") -> None

    def stop(self) -> None
```

**性能要求**：帧率 ≥ 15 FPS

---

### 模块八：CLI（命令行界面）

**文件路径**：`finalcode/cli.py`

**功能描述**：命令行交互界面

**使用方式**：
```bash
python -m finalcode.cli --mode video --input video.mp4 --output result.mp4
python -m finalcode.cli --mode camera --exercise squat
python -m finalcode.cli --mode camera --exercise pushup
```

**参数选项**：
- `--mode`：模式选择（video/camera）
- `--input`：输入视频文件路径
- `--output`：输出视频文件路径
- `--exercise`：运动类型（squat/pushup）
- `--camera-id`：摄像头设备ID
- `--conf`：置信度阈值

---

### 模块九：GUI（图形用户界面）

**文件路径**：`finalcode/gui.py`

**功能描述**：可视化交互界面

**功能要求**：
- 实时视频显示
- 骨骼可视化
- 角度数值显示
- 动作计数显示
- 标准性反馈
- 开始/停止控制
- 运动类型切换

---

## 测试要求

### 单元测试清单

| 测试模块 | 测试文件 | 测试内容 |
|----------|----------|----------|
| PoseDetector | `tests/test_pose_detector.py` | 模型加载、关键点检测、骨骼绘制 |
| AngleCalculator | `tests/test_angle_calculator.py` | 角度计算、范围判断、标准化 |
| SquatAnalyzer | `tests/test_squat_analyzer.py` | 阶段检测、计数逻辑、标准评估 |
| PushUpAnalyzer | `tests/test_pushup_analyzer.py` | 阶段检测、计数逻辑、标准评估 |
| VideoAnalyzer | `tests/test_video_analyzer.py` | 视频读取、帧处理、统计输出 |
| CameraAnalyzer | `tests/test_camera_analyzer.py` | 摄像头连接、实时处理 |

### 测试覆盖要求

1. **边界条件测试**：
   - 角度为 0°、90°、180° 的情况
   - 置信度刚好等于阈值的情况
   - 动作处于临界状态的情况

2. **异常处理测试**：
   - 未检测到人体时的处理
   - 关键点缺失时的处理
   - 视频文件无法打开的处理
   - 摄像头无法连接的处理

3. **集成测试**：
   - 完整流程测试（从视频/摄像头到结果输出）
   - 多帧连续处理测试
   - 计数准确性验证

---

## 验收标准

### 功能验收

1. [ ] YOLOv8-pose 模型成功加载
2. [ ] 17 个关键点正确检测
3. [ ] 骨骼可视化正确绘制
4. [ ] 角度计算结果准确（误差 < 1°）
5. [ ] 深蹲动作正确计数
6. [ ] 俯卧撑动作正确计数
7. [ ] 视频文件分析正常输出
8. [ ] 摄像头实时分析帧率 ≥ 15 FPS
9. [ ] CLI 命令行工具正常工作
10. [ ] GUI 界面正常显示和交互

### 代码质量验收

1. [ ] 所有模块有完整的单元测试
2. [ ] 测试覆盖率 ≥ 80%
3. [ ] 代码符合 PEP 8 规范
4. [ ] 无硬编码配置值（使用配置文件）
5. [ ] 完整的异常处理

---

## 注意事项

1. **实现顺序**：建议按以下顺序实现
   - PoseDetector → AngleCalculator → ExerciseEngine → SquatAnalyzer/PushUpAnalyzer → VideoAnalyzer/CameraAnalyzer → CLI/GUI

2. **模块独立性**：核心模块（PoseDetector、AngleCalculator）不依赖任何业务逻辑模块

3. **配置管理**：所有配置参数应集中管理，便于调整

4. **错误处理**：所有 IO 操作和模型推理必须有异常处理

5. **性能优化**：
   - 摄像头模式下使用多线程处理
   - 避免在循环中重复创建对象
   - 合理使用缓存

6. **如有疑问**：实现过程中如有不清楚的地方，请及时提出
