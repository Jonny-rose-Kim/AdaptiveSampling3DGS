"""
Trajectory Analyzer 테스트
"""

import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaptive_sampling.colmap_parser import CameraPose
from adaptive_sampling.trajectory_analyzer import TrajectoryAnalyzer


def create_linear_trajectory(n_poses: int = 5, step: float = 1.0) -> list:
    """균등한 간격으로 직선 이동하는 trajectory 생성"""
    poses = []
    for i in range(n_poses):
        pose = CameraPose(
            image_id=i + 1,
            qw=1, qx=0, qy=0, qz=0,  # no rotation
            tx=i * step, ty=0, tz=0,
            camera_id=1,
            image_name=f"frame_{i:06d}.png",
            timestamp=float(i)
        )
        poses.append(pose)
    return poses


def create_rotation_trajectory(n_poses: int = 5, angle_step: float = 45.0) -> list:
    """제자리에서 회전하는 trajectory 생성"""
    poses = []
    for i in range(n_poses):
        angle_deg = i * angle_step
        rot = Rotation.from_euler('z', angle_deg, degrees=True)
        quat = rot.as_quat()  # [x, y, z, w]

        pose = CameraPose(
            image_id=i + 1,
            qw=quat[3], qx=quat[0], qy=quat[1], qz=quat[2],
            tx=0, ty=0, tz=0,  # no translation
            camera_id=1,
            image_name=f"frame_{i:06d}.png",
            timestamp=float(i)
        )
        poses.append(pose)
    return poses


def create_mixed_trajectory() -> list:
    """이동과 회전이 혼합된 불균등 trajectory 생성"""
    poses = [
        # 작은 이동
        CameraPose(1, 1, 0, 0, 0, 0, 0, 0, 1, "frame_0.png", 0.0),
        CameraPose(2, 1, 0, 0, 0, 0.5, 0, 0, 1, "frame_1.png", 1.0),

        # 큰 이동
        CameraPose(3, 1, 0, 0, 0, 5.0, 0, 0, 1, "frame_2.png", 2.0),

        # 작은 이동 + 회전
        CameraPose(4, 0.924, 0, 0, 0.383, 5.5, 0, 0, 1, "frame_3.png", 3.0),  # ~45도

        # 작은 이동
        CameraPose(5, 0.924, 0, 0, 0.383, 6.0, 0, 0, 1, "frame_4.png", 4.0),
    ]
    return poses


def test_linear_trajectory():
    """직선 이동 trajectory 테스트"""
    print("\n=== Test: Linear Trajectory ===")

    poses = create_linear_trajectory(n_poses=5, step=1.0)
    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    # 모든 세그먼트가 동일한 거리여야 함
    assert len(segments) == 4
    print(f"✓ Created {len(segments)} segments")

    for i, seg in enumerate(segments):
        assert abs(seg.translation_distance - 1.0) < 1e-6
        assert abs(seg.rotation_distance) < 1e-6
        print(f"  Segment {i}: trans={seg.translation_distance:.3f}, "
              f"rot={np.rad2deg(seg.rotation_distance):.2f}°")

    # 누적 score 확인
    assert abs(segments[-1].cumulative_score - 4.0) < 1e-6
    print(f"✓ Total cumulative score: {segments[-1].cumulative_score:.3f}")

    stats = analyzer.get_statistics(segments)
    assert abs(stats['translation']['std']) < 1e-6  # 표준편차가 0에 가까워야 함
    print(f"✓ Translation std (should be ~0): {stats['translation']['std']:.6f}")

    print("✓ Linear trajectory test passed!")


def test_rotation_trajectory():
    """회전 trajectory 테스트"""
    print("\n=== Test: Rotation Trajectory ===")

    poses = create_rotation_trajectory(n_poses=5, angle_step=45.0)
    analyzer = TrajectoryAnalyzer(alpha=0.0, beta=1.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    assert len(segments) == 4
    print(f"✓ Created {len(segments)} segments")

    # 모든 세그먼트가 45도 회전이어야 함
    expected_angle = np.deg2rad(45.0)
    for i, seg in enumerate(segments):
        assert abs(seg.translation_distance) < 1e-6
        assert abs(seg.rotation_distance - expected_angle) < 1e-2
        print(f"  Segment {i}: trans={seg.translation_distance:.3f}, "
              f"rot={np.rad2deg(seg.rotation_distance):.2f}°")

    stats = analyzer.get_statistics(segments)
    print(f"✓ Rotation mean: {stats['rotation']['mean_degrees']:.2f}° "
          f"(expected: 45.00°)")
    assert abs(stats['rotation']['mean_degrees'] - 45.0) < 0.1

    print("✓ Rotation trajectory test passed!")


def test_mixed_trajectory():
    """혼합 trajectory 테스트"""
    print("\n=== Test: Mixed Trajectory ===")

    poses = create_mixed_trajectory()
    analyzer = TrajectoryAnalyzer(alpha=0.5, beta=0.5, normalize=True)
    segments = analyzer.analyze_trajectory(poses)

    assert len(segments) == 4
    print(f"✓ Created {len(segments)} segments")

    for i, seg in enumerate(segments):
        print(f"  Segment {i}: trans={seg.translation_distance:.3f}, "
              f"rot={np.rad2deg(seg.rotation_distance):.2f}°, "
              f"score={seg.score:.3f}")

    stats = analyzer.get_statistics(segments)
    print(f"\nStatistics:")
    print(f"  Translation: mean={stats['translation']['mean']:.3f}, "
          f"std={stats['translation']['std']:.3f}")
    print(f"  Rotation: mean={stats['rotation']['mean_degrees']:.2f}°, "
          f"std={stats['rotation']['std_degrees']:.2f}°")
    print(f"  Score: mean={stats['score']['mean']:.3f}, std={stats['score']['std']:.3f}")

    # Sparse 구간 식별
    sparse_regions = analyzer.identify_sparse_regions(segments, threshold_multiplier=1.5)
    print(f"\nSparse regions (>1.5x mean): {sparse_regions}")

    # 세그먼트 2 (큰 이동)가 sparse로 식별되어야 함
    assert len(sparse_regions) > 0
    print(f"✓ Identified {len(sparse_regions)} sparse region(s)")

    print("✓ Mixed trajectory test passed!")


def test_normalization():
    """정규화 기능 테스트"""
    print("\n=== Test: Normalization ===")

    # 큰 이동 + 작은 회전
    poses = [
        CameraPose(1, 1, 0, 0, 0, 0, 0, 0, 1, "f1.png", 0.0),
        CameraPose(2, 0.9998, 0, 0, 0.02, 100, 0, 0, 1, "f2.png", 1.0),  # 100m 이동, ~2도 회전
    ]

    # 정규화 없이 (translation이 지배적)
    analyzer_no_norm = TrajectoryAnalyzer(alpha=0.5, beta=0.5, normalize=False)
    segments_no_norm = analyzer_no_norm.analyze_trajectory(poses)

    # 정규화 사용 (translation과 rotation이 균등하게 기여)
    analyzer_norm = TrajectoryAnalyzer(alpha=0.5, beta=0.5, normalize=True)
    segments_norm = analyzer_norm.analyze_trajectory(poses)

    trans_dist = segments_no_norm[0].translation_distance
    rot_dist = segments_no_norm[0].rotation_distance

    score_no_norm = segments_no_norm[0].score
    score_norm = segments_norm[0].score

    print(f"Translation distance: {trans_dist:.3f}m")
    print(f"Rotation distance: {np.rad2deg(rot_dist):.2f}° ({rot_dist:.4f} rad)")
    print(f"\nScore without normalization: {score_no_norm:.3f}")
    print(f"Score with normalization: {score_norm:.3f}")

    # 정규화 없이는 translation이 지배적이어야 함
    assert score_no_norm > 10 * rot_dist

    # 정규화 사용 시 score는 0과 1 사이여야 함
    assert 0 <= score_norm <= 1

    print("✓ Normalization test passed!")


def test_sparse_region_detection():
    """Sparse 구간 감지 테스트"""
    print("\n=== Test: Sparse Region Detection ===")

    # 균등 - sparse - 균등 패턴
    poses = [
        CameraPose(1, 1, 0, 0, 0, 0, 0, 0, 1, "f1.png", 0.0),
        CameraPose(2, 1, 0, 0, 0, 1, 0, 0, 1, "f2.png", 1.0),
        CameraPose(3, 1, 0, 0, 0, 2, 0, 0, 1, "f3.png", 2.0),
        # sparse 구간
        CameraPose(4, 1, 0, 0, 0, 10, 0, 0, 1, "f4.png", 3.0),
        CameraPose(5, 1, 0, 0, 0, 18, 0, 0, 1, "f5.png", 4.0),
        # 다시 균등
        CameraPose(6, 1, 0, 0, 0, 19, 0, 0, 1, "f6.png", 5.0),
        CameraPose(7, 1, 0, 0, 0, 20, 0, 0, 1, "f7.png", 6.0),
    ]

    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    for i, seg in enumerate(segments):
        print(f"  Segment {i}: distance={seg.translation_distance:.1f}, "
              f"score={seg.score:.3f}")

    stats = analyzer.get_statistics(segments)
    mean_score = stats['score']['mean']
    print(f"\nMean score: {mean_score:.3f}")

    sparse_regions = analyzer.identify_sparse_regions(segments, threshold_multiplier=2.0)
    print(f"Sparse regions (>2x mean): {sparse_regions}")

    # 세그먼트 2와 3이 sparse로 감지되어야 함
    assert len(sparse_regions) == 1
    assert sparse_regions[0] == (2, 4)
    print(f"✓ Correctly identified sparse region: {sparse_regions[0]}")

    print("✓ Sparse region detection test passed!")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("Running Trajectory Analyzer Tests")
    print("="*60)

    try:
        test_linear_trajectory()
        test_rotation_trajectory()
        test_mixed_trajectory()
        test_normalization()
        test_sparse_region_detection()

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
