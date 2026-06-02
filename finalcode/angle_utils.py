import math
from typing import Tuple

class AngleCalculator:
    """角度计算工具类

    提供二维和三维空间中的角度计算功能，用于分析人体关节角度。
    """

    @staticmethod
    def calculate_angle(a: Tuple[float, float],
                        b: Tuple[float, float],
                        c: Tuple[float, float]) -> float:
        """计算由三个点形成的角度

        根据余弦定理计算点b为顶点，a-c为边的夹角。

        参数:
            a: 第一个点的坐标 (x, y)
            b: 顶点坐标 (x, y) - 角度所在位置
            c: 第三个点的坐标 (x, y)

        返回:
            夹角角度值（0-180度）
        """
        ba_x = a[0] - b[0]
        ba_y = a[1] - b[1]
        bc_x = c[0] - b[0]
        bc_y = c[1] - b[1]

        dot_product = ba_x * bc_x + ba_y * bc_y
        magnitude_ba = math.sqrt(ba_x ** 2 + ba_y ** 2)
        magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2)

        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0.0

        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        # 限值在[-1, 1]之间，避免acos函数参数超出范围
        cos_angle = max(min(cos_angle, 1.0), -1.0)

        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    @staticmethod
    def is_angle_in_range(angle: float, min_angle: float,
                          max_angle: float, tolerance: float = 5.0) -> bool:
        """判断角度是否在指定范围内

        参数:
            angle: 要检查的角度值
            min_angle: 最小角度
            max_angle: 最大角度
            tolerance: 容差(5°)

        返回:
            True表示角度在范围内，False表示超出范围
        """
        return (min_angle - tolerance) <= angle <= (max_angle + tolerance)