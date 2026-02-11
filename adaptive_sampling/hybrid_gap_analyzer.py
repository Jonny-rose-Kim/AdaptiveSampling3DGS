"""
Hybrid Gap Analyzer

Geometry와 Feature Track 연속성을 결합한 Gap Priority 계산
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    from .sfm_quality_analyzer import FeatureTrackContinuity
    from .trajectory_analyzer import TrajectorySegment
except ImportError:
    from sfm_quality_analyzer import FeatureTrackContinuity
    from trajectory_analyzer import TrajectorySegment


@dataclass
class HybridGap:
    """Hybrid 분석 기반 Gap 정보"""
    start_timestamp: float
    end_timestamp: float

    # Geometry 정보 (TrajectorySegment에서)
    camera_distance: float       # 카메라 간 물리적 거리
    geometry_score: float        # 정규화된 geometry score

    # Feature Track 정보
    shared_features: int         # 공유 feature 수
    continuity_score: float      # 정규화된 연속성 (높을수록 좋음)

    # Hybrid Priority
    gap_priority: float          # 최종 priority (높을수록 프레임 필요)

    @property
    def midpoint_timestamp(self) -> float:
        """Gap 중간 지점 timestamp"""
        return (self.start_timestamp + self.end_timestamp) / 2

    @property
    def duration(self) -> float:
        """Gap 시간 길이"""
        return self.end_timestamp - self.start_timestamp


class HybridGapAnalyzer:
    """Hybrid Gap Priority 분석기"""

    def __init__(
        self,
        geometry_weight: float = 0.5,
        continuity_weight: float = 0.5
    ):
        """
        Args:
            geometry_weight: Geometry score 가중치
            continuity_weight: Continuity score 가중치 (역수로 사용됨)
        """
        self.geometry_weight = geometry_weight
        self.continuity_weight = continuity_weight

    def compute_hybrid_gaps(
        self,
        segments: List[TrajectorySegment],
        continuities: List[FeatureTrackContinuity]
    ) -> List[HybridGap]:
        """
        Geometry와 Feature Track 정보를 결합한 Hybrid Gap 계산

        timestamp 기반 매칭을 사용하여 segment와 continuity를 연결합니다.

        Args:
            segments: TrajectoryAnalyzer의 출력
            continuities: SfMQualityAnalyzer의 Feature Track 연속성 출력

        Returns:
            HybridGap 리스트 (시간순)
        """
        # timestamp 기반 매칭을 위한 딕셔너리 생성
        # key: (start_timestamp, end_timestamp), value: segment
        seg_dict = {}
        for seg in segments:
            key = (round(seg.start_pose.timestamp, 3), round(seg.end_pose.timestamp, 3))
            seg_dict[key] = seg

        # continuity도 timestamp로 매칭
        cont_dict = {}
        for cont in continuities:
            key = (round(cont.timestamp_i, 3), round(cont.timestamp_j, 3))
            cont_dict[key] = cont

        # 매칭된 쌍 찾기
        matched_keys = set(seg_dict.keys()) & set(cont_dict.keys())

        if not matched_keys:
            # 정확한 매칭이 안 되면 근사 매칭 시도
            print("Warning: Exact timestamp matching failed, trying approximate matching...")
            matched_pairs = self._approximate_matching(segments, continuities)
        else:
            matched_pairs = [
                (seg_dict[key], cont_dict[key])
                for key in sorted(matched_keys)
            ]

        if not matched_pairs:
            raise ValueError(
                f"No matching pairs found. Segments: {len(segments)}, Continuities: {len(continuities)}"
            )

        print(f"  Matched {len(matched_pairs)} segment-continuity pairs")

        # Geometry 정규화를 위한 통계
        geo_scores = [seg.score for seg, _ in matched_pairs]
        geo_mean = np.mean(geo_scores)
        geo_std = np.std(geo_scores) + 1e-6

        # Continuity 정규화를 위한 통계
        cont_scores = [cont.continuity_score for _, cont in matched_pairs]
        cont_mean = np.mean(cont_scores)
        cont_std = np.std(cont_scores) + 1e-6

        hybrid_gaps = []

        for seg, cont in matched_pairs:
            # 정규화된 Geometry score (높을수록 카메라 많이 움직임)
            norm_geo = (seg.score - geo_mean) / geo_std

            # 정규화된 Continuity score (낮을수록 feature 연결 약함)
            norm_cont = (cont.continuity_score - cont_mean) / cont_std

            # Gap Priority = geometry 높고 + continuity 낮으면 높음
            # continuity는 역수 개념으로 사용 (낮을수록 보강 필요)
            gap_priority = (
                self.geometry_weight * norm_geo -
                self.continuity_weight * norm_cont
            )

            hybrid_gaps.append(HybridGap(
                start_timestamp=seg.start_pose.timestamp,
                end_timestamp=seg.end_pose.timestamp,
                camera_distance=seg.translation_distance,
                geometry_score=seg.score,
                shared_features=cont.shared_point_count,
                continuity_score=cont.continuity_score,
                gap_priority=gap_priority
            ))

        return hybrid_gaps

    def _approximate_matching(
        self,
        segments: List[TrajectorySegment],
        continuities: List[FeatureTrackContinuity],
        tolerance: float = 0.1
    ) -> List[Tuple[TrajectorySegment, FeatureTrackContinuity]]:
        """
        timestamp가 정확히 일치하지 않을 때 근사 매칭

        Args:
            segments: TrajectorySegment 리스트
            continuities: FeatureTrackContinuity 리스트
            tolerance: 매칭 허용 오차 (초)

        Returns:
            매칭된 (segment, continuity) 튜플 리스트
        """
        matched = []
        used_cont_indices = set()

        for seg in sorted(segments, key=lambda s: s.start_pose.timestamp):
            seg_start = seg.start_pose.timestamp
            seg_end = seg.end_pose.timestamp

            best_match = None
            best_distance = float('inf')
            best_idx = -1

            for idx, cont in enumerate(continuities):
                if idx in used_cont_indices:
                    continue

                # timestamp 거리 계산
                dist = abs(cont.timestamp_i - seg_start) + abs(cont.timestamp_j - seg_end)

                if dist < best_distance and dist < tolerance * 2:
                    best_distance = dist
                    best_match = cont
                    best_idx = idx

            if best_match is not None:
                matched.append((seg, best_match))
                used_cont_indices.add(best_idx)

        return matched

    def select_high_priority_gaps(
        self,
        gaps: List[HybridGap],
        top_k: int
    ) -> List[HybridGap]:
        """
        Priority가 높은 상위 K개 gap 선택

        Args:
            gaps: HybridGap 리스트
            top_k: 선택할 gap 수

        Returns:
            Priority 높은 순으로 정렬된 상위 K개 gap
        """
        sorted_gaps = sorted(gaps, key=lambda g: g.gap_priority, reverse=True)
        return sorted_gaps[:top_k]

    def filter_textureless_gaps(
        self,
        gaps: List[HybridGap],
        min_features_threshold: int = 50
    ) -> List[HybridGap]:
        """
        Textureless 구간 필터링

        Feature 수가 너무 적은 구간은 프레임을 추가해도 효과 없음

        Args:
            gaps: HybridGap 리스트
            min_features_threshold: 최소 공유 feature 수

        Returns:
            Textureless가 아닌 gap만 필터링
        """
        filtered = [
            gap for gap in gaps
            if gap.shared_features >= min_features_threshold
        ]

        if len(filtered) < len(gaps):
            print(f"  Filtered out {len(gaps) - len(filtered)} textureless gaps "
                  f"(threshold: {min_features_threshold})")

        return filtered

    def get_statistics(self, gaps: List[HybridGap]) -> dict:
        """Gap 통계 정보"""
        if not gaps:
            return {'num_gaps': 0}

        priorities = [g.gap_priority for g in gaps]
        distances = [g.camera_distance for g in gaps]
        continuities = [g.continuity_score for g in gaps]
        features = [g.shared_features for g in gaps]

        return {
            'num_gaps': len(gaps),
            'priority': {
                'mean': float(np.mean(priorities)),
                'std': float(np.std(priorities)),
                'min': float(np.min(priorities)),
                'max': float(np.max(priorities)),
            },
            'camera_distance': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
            },
            'continuity': {
                'mean': float(np.mean(continuities)),
                'std': float(np.std(continuities)),
                'min': float(np.min(continuities)),
                'max': float(np.max(continuities)),
            },
            'shared_features': {
                'mean': float(np.mean(features)),
                'min': int(np.min(features)),
                'max': int(np.max(features)),
            }
        }

    def print_gap_analysis(self, gaps: List[HybridGap], top_k: int = 10) -> None:
        """Gap 분석 결과 출력"""
        print("\n" + "=" * 70)
        print("HYBRID GAP ANALYSIS")
        print("=" * 70)

        stats = self.get_statistics(gaps)
        print(f"\nTotal gaps: {stats['num_gaps']}")
        print(f"Priority range: [{stats['priority']['min']:.3f}, {stats['priority']['max']:.3f}]")
        print(f"Camera distance: mean={stats['camera_distance']['mean']:.3f}m")
        print(f"Continuity: mean={stats['continuity']['mean']:.3f}")
        print(f"Shared features: mean={stats['shared_features']['mean']:.1f}, "
              f"range=[{stats['shared_features']['min']}, {stats['shared_features']['max']}]")

        # Top K high priority gaps
        print(f"\nTop {top_k} High Priority Gaps:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Time Range':<20} {'Priority':>10} {'Geo':>8} {'Cont':>8} {'Features':>10}")
        print("-" * 70)

        sorted_gaps = sorted(gaps, key=lambda g: g.gap_priority, reverse=True)
        for i, gap in enumerate(sorted_gaps[:top_k]):
            time_range = f"{gap.start_timestamp:.2f}s ~ {gap.end_timestamp:.2f}s"
            print(f"{i+1:<5} {time_range:<20} {gap.gap_priority:>10.3f} "
                  f"{gap.geometry_score:>8.3f} {gap.continuity_score:>8.3f} "
                  f"{gap.shared_features:>10}")

        print("=" * 70)


if __name__ == "__main__":
    print("HybridGapAnalyzer module - use via hybrid_pipeline.py")
