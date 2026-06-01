import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finalcode.pose_detector import PoseDetector
from finalcode.angle_utils import AngleCalculator
from finalcode.exercise.squat_analyzer import SquatAnalyzer
from finalcode.exercise.pushup_analyzer import PushUpAnalyzer
from finalcode.analyzers.video_analyzer import VideoAnalyzer
from finalcode.analyzers.camera_analyzer import CameraAnalyzer

def main():
    """命令行动作分析程序入口

    支持两种分析模式：
    - video: 分析视频文件
    - camera: 实时分析摄像头画面

    用法示例:
        python cli.py --mode camera --exercise squat
        python cli.py --mode video --input video.mp4 --output result.mp4 --exercise pushup
    """
    parser = argparse.ArgumentParser(description='人体动作分析系统')

    parser.add_argument('--mode', type=str, required=True, choices=['video', 'camera'],
                        help='分析模式: video(视频文件) 或 camera(摄像头)')
    parser.add_argument('--input', type=str, help='输入视频文件路径')
    parser.add_argument('--output', type=str, help='输出视频文件路径')
    parser.add_argument('--exercise', type=str, required=True, choices=['squat', 'pushup'],
                        help='运动类型: squat(深蹲) 或 pushup(俯卧撑)')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头设备ID')
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值 (0-1, 越低越灵敏)')

    args = parser.parse_args()

    pose_detector = PoseDetector(conf_threshold=args.conf)
    angle_calculator = AngleCalculator()

    if args.exercise == 'squat':
        analyzer = SquatAnalyzer(pose_detector, angle_calculator)
    else:
        analyzer = PushUpAnalyzer(pose_detector, angle_calculator)

    if args.mode == 'video':
        if not args.input:
            print("错误: 视频模式需要提供 --input 参数")
            sys.exit(1)

        video_analyzer = VideoAnalyzer(analyzer)
        try:
            statistics = video_analyzer.analyze_video(
                video_path=args.input,
                output_path=args.output,
                show_preview=True
            )
            print(f"\n分析完成!")
            print(f"动作次数: {statistics.total_count}")
        except Exception as e:
            print(f"视频分析失败: {e}")

    elif args.mode == 'camera':
        camera_analyzer = CameraAnalyzer(analyzer, camera_id=args.camera_id)
        try:
            print("按 'q' 退出...")
            camera_analyzer.start()
            statistics = analyzer.get_statistics()
            print(f"\n分析完成!")
            print(f"动作次数: {statistics.total_count}")
        except Exception as e:
            print(f"摄像头分析失败: {e}")

if __name__ == '__main__':
    main()