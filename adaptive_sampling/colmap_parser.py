"""
COLMAP Output Parser

COLMAP의 images.txt와 cameras.txt 파일을 파싱하여 카메라 pose 정보를 추출합니다.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import re


@dataclass
class CameraPose:
    """카메라 pose 정보를 저장하는 데이터 클래스"""
    image_id: int
    qw: float  # quaternion w (real part)
    qx: float  # quaternion x
    qy: float  # quaternion y
    qz: float  # quaternion z
    tx: float  # translation x
    ty: float  # translation y
    tz: float  # translation z
    camera_id: int
    image_name: str
    timestamp: Optional[float] = None  # 원본 비디오 timestamp (초 단위)

    @property
    def quaternion(self) -> np.ndarray:
        """Quaternion을 numpy array로 반환 (w, x, y, z)"""
        return np.array([self.qw, self.qx, self.qy, self.qz])

    @property
    def translation(self) -> np.ndarray:
        """
        COLMAP의 translation vector 반환

        주의: 이것은 world-to-camera 변환의 translation이므로
        실제 카메라 위치가 아닙니다. 실제 카메라 위치는 camera_center를 사용하세요.
        """
        return np.array([self.tx, self.ty, self.tz])

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Quaternion을 rotation matrix로 변환"""
        # COLMAP의 quaternion은 world-to-camera 변환
        # scipy는 (x, y, z, w) 순서를 사용
        rot = Rotation.from_quat([self.qx, self.qy, self.qz, self.qw])
        return rot.as_matrix()

    @property
    def camera_center(self) -> np.ndarray:
        """
        실제 카메라 위치를 월드 좌표계에서 계산

        COLMAP의 (TX, TY, TZ)는 world-to-camera 변환의 translation이므로,
        실제 카메라 중심은 C = -R^T × T 로 계산해야 합니다.

        참고:
        - COLMAP format: https://colmap.github.io/format.html
        - "The coordinates of the projection/camera center are given by -R^t * T"

        Returns:
            카메라 중심의 월드 좌표 (3,)
        """
        R = self.rotation_matrix
        T = self.translation
        return -R.T @ T


class COLMAPParser:
    """COLMAP 출력 파일을 파싱하는 클래스"""

    def __init__(self, colmap_dir: str):
        """
        Args:
        
            colmap_dir: COLMAP sparse reconstruction 디렉토리 경로
                       (보통 <dataset>/sparse/0/)
        """
        self.colmap_dir = Path(colmap_dir)
        self.images_file = self.colmap_dir / "images.txt"
        self.cameras_file = self.colmap_dir / "cameras.txt"

        if not self.images_file.exists():
            raise FileNotFoundError(f"images.txt not found: {self.images_file}")
        if not self.cameras_file.exists():
            raise FileNotFoundError(f"cameras.txt not found: {self.cameras_file}")

    def parse_images(self) -> Dict[int, CameraPose]:
        """
        images.txt 파일을 파싱하여 카메라 pose 정보를 추출합니다.

        Returns:
            image_id를 key로 하는 CameraPose 딕셔너리
        """
        poses = {}

        with open(self.images_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 주석이나 빈 줄 건너뛰기
            if not line or line.startswith('#'):
                i += 1
                continue

            # 첫 번째 줄: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            parts = line.split()
            if len(parts) < 10:
                i += 1
                continue

            try:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                image_name = parts[9]

                pose = CameraPose(
                    image_id=image_id,
                    qw=qw, qx=qx, qy=qy, qz=qz,
                    tx=tx, ty=ty, tz=tz,
                    camera_id=camera_id,
                    image_name=image_name
                )

                poses[image_id] = pose

            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line {i}: {line}")
                print(f"Error: {e}")

            # 두 번째 줄 (POINTS2D)은 건너뛰기
            i += 2

        return poses

    def extract_timestamp_from_filename(self, filename: str, fps: Optional[float] = None) -> Optional[float]:
        """
        파일명에서 timestamp를 추출합니다.

        지원하는 형식:
        - frame_000123.png -> fps 기반 계산
        - frame_000123_045.678.png -> 명시적 timestamp (45.678초)
        - 000123.png -> fps 기반 계산

        Args:
            filename: 이미지 파일명
            fps: 비디오 fps (파일명에 timestamp가 없을 때 사용)

        Returns:
            timestamp in seconds, or None if cannot extract
        """
        # 패턴 1: frame_NNNNNN_TTT.ttt.ext (명시적 timestamp)
        pattern1 = r'frame_\d+_(\d+\.\d+)\.'
        match = re.search(pattern1, filename)
        if match:
            return float(match.group(1))

        # 패턴 2: frame_NNNNNN.ext 또는 NNNNNN.ext (프레임 번호)
        pattern2 = r'(?:frame_)?(\d+)\.'
        match = re.search(pattern2, filename)
        if match and fps is not None:
            frame_number = int(match.group(1))
            return frame_number / fps

        return None

    def add_timestamps(self, poses: Dict[int, CameraPose], fps: Optional[float] = None) -> None:
        """
        파일명에서 timestamp를 추출하여 CameraPose에 추가합니다.

        Args:
            poses: parse_images()로 얻은 poses 딕셔너리
            fps: 비디오 fps (파일명에 timestamp가 없을 때 사용)
        """
        for pose in poses.values():
            timestamp = self.extract_timestamp_from_filename(pose.image_name, fps)
            pose.timestamp = timestamp

    def get_sorted_poses(self, poses: Dict[int, CameraPose]) -> List[CameraPose]:
        """
        Timestamp 순으로 정렬된 pose 리스트를 반환합니다.

        Args:
            poses: parse_images()로 얻은 poses 딕셔너리

        Returns:
            timestamp 순으로 정렬된 CameraPose 리스트
            (timestamp가 None인 경우는 제외됨)
        """
        valid_poses = [p for p in poses.values() if p.timestamp is not None]
        return sorted(valid_poses, key=lambda p: p.timestamp)

    def parse_and_extract(self, fps: Optional[float] = None) -> List[CameraPose]:
        """
        전체 파싱 프로세스를 실행합니다.

        Args:
            fps: 비디오 fps (파일명에 timestamp가 없을 때 사용)

        Returns:
            timestamp 순으로 정렬된 CameraPose 리스트
        """
        poses = self.parse_images()
        self.add_timestamps(poses, fps)
        return self.get_sorted_poses(poses)


if __name__ == "__main__":
    # 사용 예시
    import sys

    if len(sys.argv) < 2:
        print("Usage: python colmap_parser.py <colmap_dir> [fps]")
        sys.exit(1)

    colmap_dir = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else None

    parser = COLMAPParser(colmap_dir)
    poses = parser.parse_and_extract(fps)

    print(f"Parsed {len(poses)} camera poses")
    if poses:
        print(f"\nFirst pose:")
        p = poses[0]
        print(f"  Image: {p.image_name}")
        print(f"  Timestamp: {p.timestamp:.3f}s")
        print(f"  Translation: {p.translation}")
        print(f"  Quaternion: {p.quaternion}")
