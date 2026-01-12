"""
Adaptive Sampler

Trajectory 분석 결과를 바탕으로 균등한 샘플링을 위한 최적 timestamp를 계산합니다.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    from .trajectory_analyzer import TrajectorySegment
    from .colmap_parser import CameraPose
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from adaptive_sampling.trajectory_analyzer import TrajectorySegment
    from adaptive_sampling.colmap_parser import CameraPose


@dataclass
class SamplingPoint:
    """샘플링 포인트 정보"""
    target_cumulative_score: float  # 목표 누적 score
    actual_timestamp: float  # 실제 선택된 timestamp
    pose_index: int  # 선택된 pose의 인덱스
    interpolation_weight: float  # 보간 가중치 (0~1, 1이면 정확히 pose에 위치)


class AdaptiveSampler:
    """Adaptive sampling 알고리즘을 구현하는 클래스"""

    def __init__(self, video_fps: Optional[float] = None):
        """
        Args:
            video_fps: 원본 비디오 fps (가능한 timestamp를 제한하는 데 사용)
        """
        self.video_fps = video_fps

    def compute_target_timestamps(
        self,
        segments: List[TrajectorySegment],
        desired_frame_count: int
    ) -> List[float]:
        """
        균등한 score 간격으로 샘플링할 timestamp들을 계산합니다.

        Args:
            segments: TrajectorySegment 리스트
            desired_frame_count: 원하는 프레임 수

        Returns:
            timestamp 리스트 (초 단위)
        """
        if not segments or desired_frame_count < 2:
            return []

        total_score = segments[-1].cumulative_score
        target_distance = total_score / (desired_frame_count - 1)

        # 첫 프레임은 항상 포함
        timestamps = [segments[0].start_pose.timestamp]

        # 중간 프레임들
        for i in range(1, desired_frame_count - 1):
            target_score = i * target_distance
            timestamp = self._find_timestamp_at_score(segments, target_score)
            timestamps.append(timestamp)

        # 마지막 프레임은 항상 포함
        timestamps.append(segments[-1].end_pose.timestamp)

        return timestamps

    def _find_timestamp_at_score(
        self,
        segments: List[TrajectorySegment],
        target_score: float
    ) -> float:
        """
        주어진 누적 score에 해당하는 timestamp를 찾습니다.

        Args:
            segments: TrajectorySegment 리스트
            target_score: 목표 누적 score

        Returns:
            timestamp (초 단위)
        """
        # Binary search로 해당 세그먼트 찾기
        for i, seg in enumerate(segments):
            if i == 0:
                prev_cumulative = 0
            else:
                prev_cumulative = segments[i - 1].cumulative_score

            if prev_cumulative <= target_score <= seg.cumulative_score:
                # 이 세그먼트 내에서 보간
                seg_start_score = prev_cumulative
                seg_end_score = seg.cumulative_score
                seg_score_range = seg_end_score - seg_start_score

                if seg_score_range > 0:
                    # 세그먼트 내에서의 비율
                    ratio = (target_score - seg_start_score) / seg_score_range
                else:
                    ratio = 0.5

                # Timestamp 보간
                t_start = seg.start_pose.timestamp
                t_end = seg.end_pose.timestamp
                interpolated_timestamp = t_start + ratio * (t_end - t_start)

                # 가장 가까운 실제 프레임 timestamp로 스냅
                return self._snap_to_nearest_frame(interpolated_timestamp)

        # 만약 찾지 못하면 마지막 timestamp 반환
        return segments[-1].end_pose.timestamp

    def _snap_to_nearest_frame(self, timestamp: float) -> float:
        """
        주어진 timestamp를 가장 가까운 실제 프레임 timestamp로 스냅합니다.

        Args:
            timestamp: 이상적인 timestamp

        Returns:
            실제 비디오에서 추출 가능한 가장 가까운 timestamp
        """
        if self.video_fps is None:
            # FPS 정보가 없으면 그대로 반환
            return timestamp

        # 비디오 fps에 맞는 timestamp 계산
        frame_interval = 1.0 / self.video_fps
        frame_number = round(timestamp / frame_interval)
        snapped_timestamp = frame_number * frame_interval

        return snapped_timestamp

    def compute_sampling_points(
        self,
        segments: List[TrajectorySegment],
        desired_frame_count: int
    ) -> List[SamplingPoint]:
        """
        상세한 샘플링 포인트 정보를 계산합니다.

        Args:
            segments: TrajectorySegment 리스트
            desired_frame_count: 원하는 프레임 수

        Returns:
            SamplingPoint 리스트
        """
        if not segments or desired_frame_count < 2:
            return []

        # 모든 pose 리스트 생성
        poses = [segments[0].start_pose]
        for seg in segments:
            poses.append(seg.end_pose)

        total_score = segments[-1].cumulative_score
        target_distance = total_score / (desired_frame_count - 1)

        sampling_points = []

        for i in range(desired_frame_count):
            if i == 0:
                # 첫 프레임
                sampling_points.append(SamplingPoint(
                    target_cumulative_score=0.0,
                    actual_timestamp=poses[0].timestamp,
                    pose_index=0,
                    interpolation_weight=1.0
                ))
            elif i == desired_frame_count - 1:
                # 마지막 프레임
                sampling_points.append(SamplingPoint(
                    target_cumulative_score=total_score,
                    actual_timestamp=poses[-1].timestamp,
                    pose_index=len(poses) - 1,
                    interpolation_weight=1.0
                ))
            else:
                # 중간 프레임
                target_score = i * target_distance
                sp = self._find_sampling_point(segments, poses, target_score)
                sampling_points.append(sp)

        return sampling_points

    def _find_sampling_point(
        self,
        segments: List[TrajectorySegment],
        poses: List[CameraPose],
        target_score: float
    ) -> SamplingPoint:
        """
        상세한 샘플링 포인트 정보를 찾습니다.

        Args:
            segments: TrajectorySegment 리스트
            poses: 전체 CameraPose 리스트
            target_score: 목표 누적 score

        Returns:
            SamplingPoint
        """
        for i, seg in enumerate(segments):
            if i == 0:
                prev_cumulative = 0
            else:
                prev_cumulative = segments[i - 1].cumulative_score

            if prev_cumulative <= target_score <= seg.cumulative_score:
                seg_start_score = prev_cumulative
                seg_end_score = seg.cumulative_score
                seg_score_range = seg_end_score - seg_start_score

                if seg_score_range > 0:
                    ratio = (target_score - seg_start_score) / seg_score_range
                else:
                    ratio = 0.5

                t_start = seg.start_pose.timestamp
                t_end = seg.end_pose.timestamp
                interpolated_timestamp = t_start + ratio * (t_end - t_start)
                snapped_timestamp = self._snap_to_nearest_frame(interpolated_timestamp)

                # 가장 가까운 pose 찾기
                best_pose_idx = i  # start pose index
                best_diff = abs(poses[i].timestamp - snapped_timestamp)

                for j in range(len(poses)):
                    diff = abs(poses[j].timestamp - snapped_timestamp)
                    if diff < best_diff:
                        best_diff = diff
                        best_pose_idx = j

                # Interpolation weight 계산
                if best_diff < 1e-6:
                    weight = 1.0
                else:
                    weight = 1.0 - (best_diff / max(abs(t_end - t_start), 1e-6))
                    weight = max(0.0, min(1.0, weight))

                return SamplingPoint(
                    target_cumulative_score=target_score,
                    actual_timestamp=snapped_timestamp,
                    pose_index=best_pose_idx,
                    interpolation_weight=weight
                )

        # 못 찾으면 마지막 pose
        return SamplingPoint(
            target_cumulative_score=target_score,
            actual_timestamp=poses[-1].timestamp,
            pose_index=len(poses) - 1,
            interpolation_weight=1.0
        )

    def handle_sparse_regions(
        self,
        segments: List[TrajectorySegment],
        sparse_regions: List[Tuple[int, int]],
        densification_factor: int = 2
    ) -> List[float]:
        """
        Sparse 구간에 추가 샘플을 생성합니다.

        Args:
            segments: TrajectorySegment 리스트
            sparse_regions: (start_idx, end_idx) 튜플 리스트
            densification_factor: Dense 샘플링 배수

        Returns:
            추가 timestamp 리스트
        """
        additional_timestamps = []

        for start_idx, end_idx in sparse_regions:
            # Sparse 구간의 시작과 끝 timestamp
            t_start = segments[start_idx].start_pose.timestamp
            t_end = segments[end_idx - 1].end_pose.timestamp if end_idx > 0 else segments[-1].end_pose.timestamp

            # 세그먼트 수
            num_segments = end_idx - start_idx

            # 추가 샘플 수 계산
            num_additional_samples = num_segments * densification_factor

            # 균등하게 샘플링
            for i in range(1, num_additional_samples + 1):
                ratio = i / (num_additional_samples + 1)
                timestamp = t_start + ratio * (t_end - t_start)
                snapped_timestamp = self._snap_to_nearest_frame(timestamp)
                additional_timestamps.append(snapped_timestamp)

        return sorted(additional_timestamps)

    def merge_timestamps_with_limit(
        self,
        base_timestamps: List[float],
        additional_timestamps: List[float],
        segments: List[TrajectorySegment],
        desired_count: int
    ) -> List[float]:
        """
        추가 timestamps를 병합하고, 총 프레임 수를 desired_count로 유지합니다.
        초과 시 score가 낮은 구간의 프레임을 제거합니다.

        Args:
            base_timestamps: 기본 샘플링 timestamps
            additional_timestamps: sparse region에서 추가된 timestamps
            segments: trajectory segments (score 정보 포함)
            desired_count: 목표 프레임 수

        Returns:
            조정된 timestamp 리스트
        """
        # 1. 병합 및 중복 제거
        all_timestamps = sorted(set(base_timestamps + additional_timestamps))

        # 2. 목표 수 이하면 그대로 반환
        if len(all_timestamps) <= desired_count:
            return all_timestamps

        # 3. 각 timestamp의 "중요도" 계산
        #    - 해당 timestamp 주변 구간의 score가 높을수록 중요
        timestamp_scores = []
        for ts in all_timestamps:
            score = self._get_score_at_timestamp(ts, segments)
            timestamp_scores.append((ts, score))

        # 4. 첫 프레임과 마지막 프레임은 항상 유지
        first_ts = all_timestamps[0]
        last_ts = all_timestamps[-1]

        # 5. 중간 프레임들을 score 기준으로 정렬
        middle_frames = [(ts, score) for ts, score in timestamp_scores
                         if ts != first_ts and ts != last_ts]
        middle_frames.sort(key=lambda x: x[1], reverse=True)  # score 높은 순

        # 6. 상위 (desired_count - 2)개만 선택
        selected_middle = middle_frames[:desired_count - 2]
        selected_timestamps = [first_ts] + [ts for ts, _ in selected_middle] + [last_ts]

        return sorted(selected_timestamps)

    def _get_score_at_timestamp(
        self,
        timestamp: float,
        segments: List[TrajectorySegment]
    ) -> float:
        """
        주어진 timestamp가 속한 segment의 score를 반환합니다.

        Args:
            timestamp: 타임스탬프
            segments: trajectory segments

        Returns:
            해당 timestamp의 score
        """
        for seg in segments:
            if seg.start_pose.timestamp <= timestamp <= seg.end_pose.timestamp:
                return seg.score

        # 범위 밖이면 가장 가까운 segment의 score 반환
        if timestamp < segments[0].start_pose.timestamp:
            return segments[0].score
        return segments[-1].score


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Adaptive Sampler Test ===\n")

    from adaptive_sampling.colmap_parser import CameraPose
    from adaptive_sampling.trajectory_analyzer import TrajectoryAnalyzer

    # 테스트 데이터: 불균등 간격
    poses = [
        CameraPose(1, 1, 0, 0, 0, 0, 0, 0, 1, "f0.png", 0.0),
        CameraPose(2, 1, 0, 0, 0, 0.5, 0, 0, 1, "f1.png", 1.0),
        CameraPose(3, 1, 0, 0, 0, 5.0, 0, 0, 1, "f2.png", 2.0),  # 큰 간격
        CameraPose(4, 1, 0, 0, 0, 5.5, 0, 0, 1, "f3.png", 3.0),
        CameraPose(5, 1, 0, 0, 0, 6.0, 0, 0, 1, "f4.png", 4.0),
    ]

    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    print("Original trajectory:")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: {seg.start_pose.timestamp:.1f}s -> {seg.end_pose.timestamp:.1f}s, "
              f"dist={seg.translation_distance:.2f}, score={seg.score:.2f}")

    sampler = AdaptiveSampler(video_fps=30.0)

    # 3개 프레임으로 리샘플링
    timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=5)

    print(f"\nAdaptive sampling (5 frames):")
    for i, ts in enumerate(timestamps):
        print(f"  Frame {i}: {ts:.3f}s")

    # Sampling points with details
    sampling_points = sampler.compute_sampling_points(segments, desired_frame_count=5)

    print(f"\nDetailed sampling points:")
    for i, sp in enumerate(sampling_points):
        print(f"  Point {i}: target_score={sp.target_cumulative_score:.2f}, "
              f"timestamp={sp.actual_timestamp:.3f}s, "
              f"pose_idx={sp.pose_index}, weight={sp.interpolation_weight:.2f}")

    print("\n✓ Adaptive Sampler basic test completed!")
