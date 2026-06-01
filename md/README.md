# 人体动作分析系统（Human Pose Analysis System）

基于 YOLOv8-pose 姿态检测的实时运动分析系统，支持**深蹲**和**俯卧撑**两种运动的动作计数与标准性评估。

## 功能特性

- 🎯 **实时姿态检测**：基于 YOLOv8-pose 的 17 点人体关键点检测
- 🏃 **深蹲分析**：膝盖角度、髋部角度检测，四阶段状态机（站立→下蹲→底部→起身）
- 💪 **俯卧撑分析**：肘部角度、身体直线度检测，四阶段状态机（撑起→下降→底部→推起）
- 📐 **自适应校准**：前 30 帧自动学习用户站立/撑起姿态，计算个性化阈值
- 🖥️ **GUI 界面**：PyQt5 图形界面，实时骨骼可视化、角度显示、反馈提示
- 🎬 **命令行工具**：支持摄像头实时分析和视频文件离线分析
- 📝 **中文反馈**：全程中文界面与动作建议提示

## 技术栈

| 组件     | 技术                      |
| -------- | ------------------------- |
| 编程语言 | Python 3.8+               |
| 姿态检测 | YOLOv8-pose (ultralytics) |
| 深度学习 | PyTorch                   |
| 图像处理 | OpenCV, PIL (Pillow)      |
| GUI      | PyQt5                     |

## 安装

```bash
# 克隆项目后进入目录
cd fincode

# 安装依赖
pip install -r requirements.txt
```

模型文件 `yolov8s-pose.pt` 和 `yolov8n-pose.pt` 已包含在项目中。

## 项目结构

```
fincode/
├── finalcode/
│   ├── __init__.py              # 包入口
│   ├── pose_detector.py         # 姿态检测器（YOLOv8-pose 封装）
│   ├── angle_utils.py           # 角度计算工具（2D/3D）
│   ├── gui.py                   # PyQt5 图形用户界面
│   ├── cli.py                   # 命令行界面
│   ├── yolov8n-pose.pt          # YOLOv8 nano-pose 模型
│   ├── exercise/
│   │   ├── __init__.py
│   │   ├── engine.py            # 运动分析引擎抽象基类
│   │   ├── squat_analyzer.py    # 深蹲分析器
│   │   └── pushup_analyzer.py   # 俯卧撑分析器
│   └── analyzers/
│       ├── __init__.py
│       ├── video_analyzer.py    # 视频文件分析器
│       └── camera_analyzer.py   # 摄像头实时分析器
├── video/
│   ├── input/                   # 测试输入视频
│   └── output/                  # 分析输出视频
├── finalcode/yolov8n-pose.pt    # YOLOv8 nano 模型（轻量）
├── yolov8s-pose.pt              # YOLOv8 small 模型（默认）
├── requirements.txt             # Python 依赖
└── README.md
```

## 使用方式

### 1. GUI 图形界面（推荐）

```bash
cd fincode
python -m finalcode.gui
```

- 选择运动类型（深蹲/俯卧撑）
- 点击"开始"进行摄像头实时分析
- 左侧显示骨骼可视化画面，右侧显示角度和反馈

### 2. 命令行 — 摄像头实时分析

```bash
# 深蹲
python -m finalcode.cli --mode camera --exercise squat

# 俯卧撑
python -m finalcode.cli --mode camera --exercise pushup

# 自定义摄像头和置信度
python -m finalcode.cli --mode camera --exercise squat --camera-id 1 --conf 0.3
```

按 `q` 退出分析。

### 3. 命令行 — 视频文件分析

```bash
# 分析视频并保存结果
# 分析视频并保存结果
python -m finalcode.cli --mode video --exercise squat --input video/input/squat.mp4 --output result.mp4

# 分析俯卧撑视频
python -m finalcode.cli --mode video --exercise pushup --input video/input/pushup.mp4 --output result.mp4
```

### 4. 代码调用

```python
from finalcode.pose_detector import PoseDetector
from finalcode.angle_utils import AngleCalculator
from finalcode.exercise.squat_analyzer import SquatAnalyzer
from finalcode.analyzers.video_analyzer import VideoAnalyzer

detector = PoseDetector()
ac = AngleCalculator()
analyzer = SquatAnalyzer(detector, ac)

# 分析视频文件
va = VideoAnalyzer(analyzer)
stats = va.analyze_video("test.mp4", output_path="result.mp4")
print(f"动作次数: {stats.total_count}, 标准率: {stats.standard_rate:.1f}%")
```

## 参数说明

| 参数          | 说明                         | 默认值 |
| ------------- | ---------------------------- | ------ |
| `--mode`      | 分析模式：`camera` / `video` | 必填   |
| `--exercise`  | 运动类型：`squat` / `pushup` | 必填   |
| `--input`     | 输入视频路径（video 模式）   | -      |
| `--output`    | 输出视频路径                 | -      |
| `--camera-id` | 摄像头 ID                    | 0      |
| `--conf`      | 检测置信度阈值 (0-1)         | 0.3    |

## 评估标准

### 深蹲
- **膝盖角度**：站立 155-180°，底部 70-110°
- **髋部角度**：站立 155-180°，底部 40-80°
- 系统自动校准个性化阈值

### 俯卧撑
- **肘部角度**：撑起 150-180°，底部 60-100°
- **身体直线度**：165-180°（不塌腰、不拱背）
- 系统自动校准个性化阈值

## 许可证

本项目仅用于学习和研究目的。
