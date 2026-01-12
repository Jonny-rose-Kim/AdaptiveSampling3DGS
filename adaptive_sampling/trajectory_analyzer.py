"""
Trajectory Analyzer

카메라 trajectory를 분석하고 translation/rotation 기반 거리 메트릭을 계산합니다.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .colmap_parser import CameraPose
    from .utils.rotation_utils import rotation_geodesic_distance, rotation_angle_from_quaternions
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from adaptive_sampling.colmap_parser import CameraPose
    from adaptive_sampling.utils.rotation_utils import rotation_geodesic_distance, rotation_angle_from_quaternions


@dataclass
class TrajectorySegment:
    """연속된 두 카메라 pose 사이의 세그먼트 정보"""
    start_pose: CameraPose
    end_pose: CameraPose
    translation_distance: float  # Euclidean distance
    rotation_distance: float  # Geodesic distance in radians
    score: float  # Weighted combination of translation and rotation
    cumulative_score: float  # Cumulative score up to this segment


class TrajectoryAnalyzer:
    """카메라 trajectory 분석 클래스"""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, normalize: bool = True):
        """
        Args:
            alpha: Translation distance의 가중치
            beta: Rotation distance의 가중치
            normalize: True인 경우 translation과 rotation을 정규화하여 균등하게 비교
        """
        self.alpha = alpha
        self.beta = beta
        self.normalize = normalize

    def compute_translation_distance(self, pose1: CameraPose, pose2: CameraPose) -> float:
        """
        두 pose 간의 translation distance (Euclidean distance) 계산

        실제 카메라 중심 간의 거리를 계산합니다.
        COLMAP의 translation vector가 아닌 camera_center를 사용합니다.

        Args:
            pose1: 첫 번째 카메라 pose
            pose2: 두 번째 카메라 pose

        Returns:
            실제 카메라 중심 간의 Euclidean distance (미터)
        """
        return np.linalg.norm(pose2.camera_center - pose1.camera_center)

    def compute_rotation_distance(self, pose1: CameraPose, pose2: CameraPose) -> float:
        """
        두 pose 간의 rotation distance (geodesic distance) 계산

        Args:
            pose1: 첫 번째 카메라 pose
            pose2: 두 번째 카메라 pose

        Returns:
            각도 거리 (radians)
        """
        return rotation_angle_from_quaternions(pose1.quaternion, pose2.quaternion)

    def compute_score(
        self,
        trans_dist: float,
        rot_dist: float,
        trans_normalizer: float = 1.0,
        rot_normalizer: float = 1.0
    ) -> float:
        """
        Translation과 rotation distance를 결합한 score 계산

        Args:
            trans_dist: Translation distance
            rot_dist: Rotation distance
            trans_normalizer: Translation 정규화 값 (95th percentile 등)
            rot_normalizer: Rotation 정규화 값 (95th percentile 등)

        Returns:
            Combined score
        """
        if self.normalize:
            trans_normalized = trans_dist / trans_normalizer if trans_normalizer > 0 else 0
            rot_normalized = rot_dist / rot_normalizer if rot_normalizer > 0 else 0

            # Clipping: 정규화 값이 1.0을 초과하지 않도록 제한
            # 이상치가 과도한 영향을 미치지 않도록 함
            trans_normalized = min(trans_normalized, 1.0)
            rot_normalized = min(rot_normalized, 1.0)

            return self.alpha * trans_normalized + self.beta * rot_normalized
        else:
            return self.alpha * trans_dist + self.beta * rot_dist

    def analyze_trajectory(self, poses: List[CameraPose]) -> List[TrajectorySegment]:
        """
        전체 trajectory를 분석하여 세그먼트 리스트를 생성

        Args:
            poses: timestamp 순으로 정렬된 CameraPose 리스트

        Returns:
            TrajectorySegment 리스트 (길이는 len(poses) - 1)
        """
        if len(poses) < 2:
            return []

        # 1단계: 모든 거리 계산
        trans_dists = []
        rot_dists = []

        for i in range(len(poses) - 1):
            trans_dist = self.compute_translation_distance(poses[i], poses[i + 1])
            rot_dist = self.compute_rotation_distance(poses[i], poses[i + 1])
            trans_dists.append(trans_dist)
            rot_dists.append(rot_dist)

        # 2단계: 정규화를 위한 95th percentile 계산 (이상치에 강건함)
        # max() 대신 percentile을 사용하여 이상치의 영향을 제한
        trans_normalizer = np.percentile(trans_dists, 95) if trans_dists else 1.0
        rot_normalizer = np.percentile(rot_dists, 95) if rot_dists else 1.0

        # 0으로 나누기 방지
        trans_normalizer = max(trans_normalizer, 1e-6)
        rot_normalizer = max(rot_normalizer, 1e-6)

        # 3단계: Score 계산 및 세그먼트 생성
        segments = []
        cumulative_score = 0.0

        for i in range(len(poses) - 1):
            score = self.compute_score(
                trans_dists[i],
                rot_dists[i],
                trans_normalizer,
                rot_normalizer
            )
            cumulative_score += score

            segment = TrajectorySegment(
                start_pose=poses[i],
                end_pose=poses[i + 1],
                translation_distance=trans_dists[i],
                rotation_distance=rot_dists[i],
                score=score,
                cumulative_score=cumulative_score
            )
            segments.append(segment)

        return segments

    def get_statistics(self, segments: List[TrajectorySegment]) -> dict:
        """
        Trajectory 세그먼트의 통계 정보를 계산

        Args:
            segments: TrajectorySegment 리스트

        Returns:
            통계 정보 딕셔너리
        """
        if not segments:
            return {}

        trans_dists = [s.translation_distance for s in segments]
        rot_dists = [s.rotation_distance for s in segments]
        scores = [s.score for s in segments]

        total_score = segments[-1].cumulative_score if segments else 0.0

        stats = {
            'num_segments': len(segments),
            'total_score': total_score,
            'translation': {
                'mean': np.mean(trans_dists),
                'std': np.std(trans_dists),
                'min': np.min(trans_dists),
                'max': np.max(trans_dists),
            },
            'rotation': {
                'mean': np.mean(rot_dists),
                'mean_degrees': np.rad2deg(np.mean(rot_dists)),
                'std': np.std(rot_dists),
                'std_degrees': np.rad2deg(np.std(rot_dists)),
                'min': np.min(rot_dists),
                'min_degrees': np.rad2deg(np.min(rot_dists)),
                'max': np.max(rot_dists),
                'max_degrees': np.rad2deg(np.max(rot_dists)),
            },
            'score': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
            }
        }

        return stats

    def identify_sparse_regions(
        self,
        segments: List[TrajectorySegment],
        threshold_multiplier: float = 2.0
    ) -> List[Tuple[int, int]]:
        """
        Score가 평균의 threshold_multiplier 배 이상인 sparse 구간을 식별

        Args:
            segments: TrajectorySegment 리스트
            threshold_multiplier: 평균 대비 배수 (예: 2.0 = 평균의 2배)

        Returns:
            (start_index, end_index) 튜플 리스트
        """
        if not segments:
            return []

        scores = [s.score for s in segments]
        mean_score = np.mean(scores)
        threshold = mean_score * threshold_multiplier

        sparse_regions = []
        in_sparse = False
        start_idx = 0

        for i, segment in enumerate(segments):
            if segment.score >= threshold:
                if not in_sparse:
                    start_idx = i
                    in_sparse = True
            else:
                if in_sparse:
                    sparse_regions.append((start_idx, i))
                    in_sparse = False

        # 마지막이 sparse 구간이면 추가
        if in_sparse:
            sparse_regions.append((start_idx, len(segments)))

        return sparse_regions


if __name__ == "__main__":
    # 간단한 테스트
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from adaptive_sampling.colmap_parser import CameraPose

    print("=== Trajectory Analyzer Test ===\n")

    # 테스트 데이터 생성: 직선 이동 + 회전
    poses = [
        CameraPose(1, 1, 0, 0, 0, 0, 0, 0, 1, "frame_0.png", 0.0),
        CameraPose(2, 1, 0, 0, 0, 1, 0, 0, 1, "frame_1.png", 1.0),
        CameraPose(3, 1, 0, 0, 0, 2, 0, 0, 1, "frame_2.png", 2.0),
        # 큰 이동
        CameraPose(4, 1, 0, 0, 0, 5, 0, 0, 1, "frame_3.png", 3.0),
        CameraPose(5, 1, 0, 0, 0, 6, 0, 0, 1, "frame_4.png", 4.0),
    ]

    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=True)
    segments = analyzer.analyze_trajectory(poses)

    print(f"Total segments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"Segment {i}: trans={seg.translation_distance:.3f}, "
              f"rot={np.rad2deg(seg.rotation_distance):.2f}°, "
              f"score={seg.score:.3f}, cumulative={seg.cumulative_score:.3f}")

    stats = analyzer.get_statistics(segments)
    print(f"\nStatistics:")
    print(f"  Total score: {stats['total_score']:.3f}")
    print(f"  Translation: mean={stats['translation']['mean']:.3f}, "
          f"std={stats['translation']['std']:.3f}")
    print(f"  Rotation: mean={stats['rotation']['mean_degrees']:.2f}°, "
          f"std={stats['rotation']['std_degrees']:.2f}°")

    sparse_regions = analyzer.identify_sparse_regions(segments, threshold_multiplier=1.5)
    print(f"\nSparse regions (threshold=1.5x mean): {sparse_regions}")

    print("\n✓ Trajectory Analyzer basic test completed!")
