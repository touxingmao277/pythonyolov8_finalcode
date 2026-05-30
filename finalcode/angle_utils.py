import math
from typing import Tuple

class AngleCalculator:
    @staticmethod
    def calculate_angle(a: Tuple[float, float],
                        b: Tuple[float, float],
                        c: Tuple[float, float]) -> float:
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
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    @staticmethod
    def calculate_angle_3d(a: Tuple[float, float, float],
                           b: Tuple[float, float, float],
                           c: Tuple[float, float, float]) -> float:
        ba_x = a[0] - b[0]
        ba_y = a[1] - b[1]
        ba_z = a[2] - b[2]
        bc_x = c[0] - b[0]
        bc_y = c[1] - b[1]
        bc_z = c[2] - b[2]
        
        dot_product = ba_x * bc_x + ba_y * bc_y + ba_z * bc_z
        magnitude_ba = math.sqrt(ba_x ** 2 + ba_y ** 2 + ba_z ** 2)
        magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2 + bc_z ** 2)
        
        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0.0
        
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    @staticmethod
    def is_angle_in_range(angle: float, min_angle: float,
                          max_angle: float, tolerance: float = 5.0) -> bool:
        return (min_angle - tolerance) <= angle <= (max_angle + tolerance)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        angle = angle % 360.0
        if angle > 180.0:
            angle -= 360.0
        return angle