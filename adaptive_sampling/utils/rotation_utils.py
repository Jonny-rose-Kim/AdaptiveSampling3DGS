"""
Rotation 관련 유틸리티 함수들
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Quaternion을 rotation matrix로 변환

    Args:
        quat: [w, x, y, z] 형식의 quaternion

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    # scipy는 (x, y, z, w) 순서를 사용
    rot = Rotation.from_quat([x, y, z, w])
    return rot.as_matrix()


def rotation_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    두 rotation matrix 간의 geodesic distance (각도 거리)를 계산

    Geodesic distance는 SO(3) 매니폴드에서의 최단 거리로,
    두 rotation을 연결하는 최소 회전 각도입니다.

    Args:
        R1: 첫 번째 3x3 rotation matrix
        R2: 두 번째 3x3 rotation matrix

    Returns:
        각도 거리 (radians)
    """
    # R_rel = R2 * R1^T (relative rotation)
    R_rel = R2 @ R1.T

    # Rotation matrix를 axis-angle로 변환하여 각도 추출
    rot_rel = Rotation.from_matrix(R_rel)

    # Magnitude of rotation (in radians)
    angle = rot_rel.magnitude()

    return angle


def rotation_angle_from_quaternions(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    두 quaternion 간의 각도 거리를 계산 (더 효율적인 방법)

    Args:
        q1: [w, x, y, z] 형식의 첫 번째 quaternion
        q2: [w, x, y, z] 형식의 두 번째 quaternion

    Returns:
        각도 거리 (radians)
    """
    # scipy 형식으로 변환: [x, y, z, w]
    q1_scipy = np.array([q1[1], q1[2], q1[3], q1[0]])
    q2_scipy = np.array([q2[1], q2[2], q2[3], q2[0]])

    rot1 = Rotation.from_quat(q1_scipy)
    rot2 = Rotation.from_quat(q2_scipy)

    # Relative rotation
    rot_rel = rot2 * rot1.inv()

    return rot_rel.magnitude()


if __name__ == "__main__":
    # 테스트
    print("=== Rotation Utils Test ===\n")

    # 테스트 1: 90도 회전
    R1 = np.eye(3)
    R2 = Rotation.from_euler('z', 90, degrees=True).as_matrix()

    angle = rotation_geodesic_distance(R1, R2)
    print(f"90 degree rotation: {np.rad2deg(angle):.2f} degrees")
    assert abs(np.rad2deg(angle) - 90.0) < 0.01

    # 테스트 2: 180도 회전
    R3 = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    angle2 = rotation_geodesic_distance(R1, R3)
    print(f"180 degree rotation: {np.rad2deg(angle2):.2f} degrees")
    assert abs(np.rad2deg(angle2) - 180.0) < 0.01

    # 테스트 3: Quaternion 방법과 비교
    q1 = np.array([1, 0, 0, 0])  # identity
    q2 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])  # 90 degree around z

    angle3 = rotation_angle_from_quaternions(q1, q2)
    print(f"90 degree rotation (quaternion): {np.rad2deg(angle3):.2f} degrees")
    assert abs(np.rad2deg(angle3) - 90.0) < 0.01

    print("\n✓ All rotation utils tests passed!")
