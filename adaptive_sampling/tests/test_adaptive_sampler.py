"""
Adaptive Sampler 테스트
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaptive_sampling.colmap_parser import CameraPose
from adaptive_sampling.trajectory_analyzer import TrajectoryAnalyzer
from adaptive_sampling.adaptive_sampler import AdaptiveSampler


def create_uniform_trajectory(n_poses: int = 10, step: float = 1.0) -> list:
    """균등한 간격의 trajectory"""
    poses = []
    for i in range(n_poses):
        pose = CameraPose(
            image_id=i + 1,
            qw=1, qx=0, qy=0, qz=0,
            tx=i * step, ty=0, tz=0,
            camera_id=1,
            image_name=f"frame_{i:06d}.png",
            timestamp=float(i)
        )
        poses.append(pose)
    return poses


def create_sparse_trajectory() -> list:
    """Sparse 구간이 있는 trajectory"""
    poses = [
        CameraPose(1, 1, 0, 0, 0, 0, 0, 0, 1, "f0.png", 0.0),
        CameraPose(2, 1, 0, 0, 0, 1, 0, 0, 1, "f1.png", 1.0),
        CameraPose(3, 1, 0, 0, 0, 2, 0, 0, 1, "f2.png", 2.0),
        # Sparse 구간 (큰 이동)
        CameraPose(4, 1, 0, 0, 0, 10, 0, 0, 1, "f3.png", 3.0),
        CameraPose(5, 1, 0, 0, 0, 11, 0, 0, 1, "f4.png", 4.0),
    ]
    return poses


def test_uniform_sampling():
    """균등한 trajectory에서의 샘플링 테스트"""
    print("\n=== Test: Uniform Sampling ===")

    poses = create_uniform_trajectory(n_poses=10, step=1.0)
    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    sampler = AdaptiveSampler(video_fps=None)  # No FPS constraint

    # 5개 프레임으로 샘플링 (0, 2.25, 4.5, 6.75, 9)
    timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=5)

    print(f"Sampled {len(timestamps)} timestamps:")
    for i, ts in enumerate(timestamps):
        print(f"  Frame {i}: {ts:.3f}s")

    # 첫 프레임과 마지막 프레임 확인
    assert len(timestamps) == 5
    assert timestamps[0] == 0.0
    assert timestamps[-1] == 9.0

    # 간격이 대략 균등한지 확인
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    print(f"  Intervals: mean={mean_interval:.3f}, std={std_interval:.3f}")

    # 표준편차가 작아야 함
    assert std_interval < 0.5

    print("✓ Uniform sampling test passed!")


def test_sparse_region_sampling():
    """Sparse 구간 샘플링 테스트"""
    print("\n=== Test: Sparse Region Sampling ===")

    poses = create_sparse_trajectory()
    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    print("Original segments:")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: {seg.start_pose.timestamp:.1f}s -> {seg.end_pose.timestamp:.1f}s, "
              f"dist={seg.translation_distance:.1f}")

    sampler = AdaptiveSampler(video_fps=None)

    # 5개 프레임으로 샘플링
    timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=5)

    print(f"\nSampled {len(timestamps)} timestamps:")
    for i, ts in enumerate(timestamps):
        print(f"  Frame {i}: {ts:.3f}s")

    assert len(timestamps) == 5
    assert timestamps[0] == 0.0
    assert timestamps[-1] == 4.0

    # Sparse 구간(2.0~3.0)에서 대부분의 샘플이 나와야 함
    samples_in_sparse = sum(1 for ts in timestamps if 2.0 < ts < 3.0)
    print(f"  Samples in sparse region [2.0, 3.0]: {samples_in_sparse}")

    # 적어도 1개 이상의 샘플이 sparse 구간에 있어야 함
    assert samples_in_sparse >= 1

    print("✓ Sparse region sampling test passed!")


def test_fps_snapping():
    """FPS 기반 timestamp snapping 테스트"""
    print("\n=== Test: FPS Snapping ===")

    poses = create_uniform_trajectory(n_poses=5, step=1.0)
    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    # 30fps로 제한
    sampler = AdaptiveSampler(video_fps=30.0)
    timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=5)

    print(f"Sampled timestamps (30fps):")
    for i, ts in enumerate(timestamps):
        frame_number = ts * 30.0
        print(f"  Frame {i}: {ts:.6f}s (frame #{frame_number:.2f})")

        # 30fps에 정확히 맞는 timestamp인지 확인
        expected_frame = round(frame_number)
        expected_ts = expected_frame / 30.0
        diff = abs(ts - expected_ts)

        # 허용 오차 내에 있어야 함
        assert diff < 1e-6, f"Timestamp {ts} not aligned to 30fps (diff={diff})"

    print("✓ FPS snapping test passed!")


def test_sampling_points():
    """상세한 sampling points 테스트"""
    print("\n=== Test: Sampling Points ===")

    poses = create_sparse_trajectory()
    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    sampler = AdaptiveSampler(video_fps=None)
    sampling_points = sampler.compute_sampling_points(segments, desired_frame_count=5)

    print(f"Sampling points:")
    for i, sp in enumerate(sampling_points):
        print(f"  Point {i}: score={sp.target_cumulative_score:.2f}, "
              f"ts={sp.actual_timestamp:.3f}s, "
              f"pose_idx={sp.pose_index}, weight={sp.interpolation_weight:.2f}")

    assert len(sampling_points) == 5

    # 첫 번째와 마지막은 정확히 pose에 위치해야 함
    assert sampling_points[0].interpolation_weight == 1.0
    assert sampling_points[-1].interpolation_weight == 1.0

    # 모든 포인트가 유효한 timestamp를 가져야 함
    for sp in sampling_points:
        assert sp.actual_timestamp >= 0.0
        assert 0 <= sp.pose_index < len(poses)
        assert 0.0 <= sp.interpolation_weight <= 1.0

    print("✓ Sampling points test passed!")


def test_handle_sparse_regions():
    """Sparse 구간 추가 샘플링 테스트"""
    print("\n=== Test: Handle Sparse Regions ===")

    poses = create_sparse_trajectory()
    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    # Sparse 구간 식별
    sparse_regions = analyzer.identify_sparse_regions(segments, threshold_multiplier=2.0)
    print(f"Sparse regions: {sparse_regions}")

    sampler = AdaptiveSampler(video_fps=None)

    # Sparse 구간에 추가 샘플 생성 (2배 densification)
    additional_timestamps = sampler.handle_sparse_regions(
        segments,
        sparse_regions,
        densification_factor=2
    )

    print(f"Additional timestamps for sparse regions:")
    for ts in additional_timestamps:
        print(f"  {ts:.3f}s")

    # 추가 샘플이 생성되었는지 확인
    assert len(additional_timestamps) > 0

    # 모든 추가 샘플이 sparse 구간 내에 있는지 확인
    for start_idx, end_idx in sparse_regions:
        t_start = segments[start_idx].start_pose.timestamp
        t_end = segments[end_idx - 1].end_pose.timestamp

        samples_in_region = [ts for ts in additional_timestamps if t_start <= ts <= t_end]
        print(f"  Samples in region [{t_start}, {t_end}]: {len(samples_in_region)}")

        assert len(samples_in_region) > 0

    print("✓ Handle sparse regions test passed!")


def test_edge_cases():
    """Edge case 테스트"""
    print("\n=== Test: Edge Cases ===")

    # Case 1: 프레임이 2개만 있는 경우
    poses = create_uniform_trajectory(n_poses=2)
    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    sampler = AdaptiveSampler()
    timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=2)

    assert len(timestamps) == 2
    assert timestamps[0] == 0.0
    assert timestamps[1] == 1.0
    print("✓ Case 1 (2 frames): passed")

    # Case 2: desired_frame_count가 원본보다 많은 경우
    poses = create_uniform_trajectory(n_poses=3)
    segments = analyzer.analyze_trajectory(poses)

    timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=10)
    assert len(timestamps) == 10
    print("✓ Case 2 (more frames than original): passed")

    # Case 3: 빈 세그먼트
    timestamps = sampler.compute_target_timestamps([], desired_frame_count=5)
    assert len(timestamps) == 0
    print("✓ Case 3 (empty segments): passed")

    print("✓ Edge cases test passed!")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("Running Adaptive Sampler Tests")
    print("="*60)

    try:
        test_uniform_sampling()
        test_sparse_region_sampling()
        test_fps_snapping()
        test_sampling_points()
        test_handle_sparse_regions()
        test_edge_cases()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
