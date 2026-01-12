"""
Adaptive Sampling 사용 예시

이 스크립트는 adaptive sampling의 기본 사용법을 보여줍니다.
"""

import sys
from pathlib import Path

# Add adaptive_sampling to path
sys.path.insert(0, str(Path(__file__).parent))

from adaptive_sampling.colmap_parser import CameraPose
from adaptive_sampling.trajectory_analyzer import TrajectoryAnalyzer
from adaptive_sampling.adaptive_sampler import AdaptiveSampler
from adaptive_sampling.config import AdaptiveSamplingConfig


def example_1_basic_analysis():
    """
    예시 1: 기본 trajectory 분석
    """
    print("\n" + "="*70)
    print("예시 1: 기본 Trajectory 분석")
    print("="*70)

    # 샘플 카메라 pose 데이터 (실제로는 COLMAP에서 파싱)
    poses = [
        CameraPose(1, 1, 0, 0, 0, 0, 0, 0, 1, "frame_00.png", 0.0),
        CameraPose(2, 1, 0, 0, 0, 1, 0, 0, 1, "frame_10.png", 0.333),
        CameraPose(3, 1, 0, 0, 0, 2, 0, 0, 1, "frame_20.png", 0.667),
        # 큰 이동 (sparse 구간)
        CameraPose(4, 1, 0, 0, 0, 10, 0, 0, 1, "frame_30.png", 1.0),
        CameraPose(5, 1, 0, 0, 0, 11, 0, 0, 1, "frame_40.png", 1.333),
        CameraPose(6, 1, 0, 0, 0, 12, 0, 0, 1, "frame_50.png", 1.667),
    ]

    print(f"\n입력: {len(poses)}개의 카메라 pose")

    # Trajectory 분석
    analyzer = TrajectoryAnalyzer(alpha=0.5, beta=0.5, normalize=True)
    segments = analyzer.analyze_trajectory(poses)

    print(f"\n분석 결과:")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: "
              f"{seg.start_pose.timestamp:.3f}s -> {seg.end_pose.timestamp:.3f}s, "
              f"distance={seg.translation_distance:.2f}, "
              f"score={seg.score:.3f}")

    # 통계
    stats = analyzer.get_statistics(segments)
    print(f"\n통계:")
    print(f"  Total score: {stats['total_score']:.3f}")
    print(f"  Translation: mean={stats['translation']['mean']:.3f}, "
          f"std={stats['translation']['std']:.3f}")

    # Sparse 구간 식별
    sparse_regions = analyzer.identify_sparse_regions(segments, threshold_multiplier=2.0)
    print(f"\nSparse 구간 (>2x mean): {sparse_regions}")


def example_2_adaptive_sampling():
    """
    예시 2: Adaptive sampling으로 프레임 선택
    """
    print("\n" + "="*70)
    print("예시 2: Adaptive Sampling")
    print("="*70)

    # 불균등 trajectory
    poses = [
        CameraPose(1, 1, 0, 0, 0, 0, 0, 0, 1, "f0.png", 0.0),
        CameraPose(2, 1, 0, 0, 0, 0.5, 0, 0, 1, "f1.png", 0.5),
        CameraPose(3, 1, 0, 0, 0, 1, 0, 0, 1, "f2.png", 1.0),
        # Sparse 구간
        CameraPose(4, 1, 0, 0, 0, 10, 0, 0, 1, "f3.png", 2.0),
        CameraPose(5, 1, 0, 0, 0, 10.5, 0, 0, 1, "f4.png", 2.5),
        CameraPose(6, 1, 0, 0, 0, 11, 0, 0, 1, "f5.png", 3.0),
    ]

    print(f"\n원본 trajectory: {len(poses)}개 프레임")
    for p in poses:
        print(f"  {p.image_name}: t={p.timestamp:.2f}s, pos=({p.tx:.1f}, {p.ty:.1f}, {p.tz:.1f})")

    # Trajectory 분석
    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    # Adaptive sampling
    sampler = AdaptiveSampler(video_fps=None)  # FPS 제한 없음
    timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=5)

    print(f"\nAdaptive sampling 결과: {len(timestamps)}개 프레임")
    for i, ts in enumerate(timestamps):
        print(f"  Frame {i}: t={ts:.3f}s")

    # Sampling points 상세 정보
    sampling_points = sampler.compute_sampling_points(segments, desired_frame_count=5)

    print(f"\n상세 샘플링 포인트:")
    for i, sp in enumerate(sampling_points):
        print(f"  Point {i}: "
              f"target_score={sp.target_cumulative_score:.2f}, "
              f"timestamp={sp.actual_timestamp:.3f}s, "
              f"pose_idx={sp.pose_index}")


def example_3_config():
    """
    예시 3: 설정 파일 사용
    """
    print("\n" + "="*70)
    print("예시 3: 설정 파일 사용")
    print("="*70)

    # 기본 설정
    config = AdaptiveSamplingConfig.get_default()
    print(f"\n기본 설정:")
    print(config)

    # 커스텀 설정
    custom_config = AdaptiveSamplingConfig(
        alpha=0.6,
        beta=0.4,
        normalize=True,
        sparse_threshold_multiplier=2.5,
        pass1_fps=30.0
    )

    print(f"\n커스텀 설정:")
    print(custom_config)

    # 설정 저장
    config_file = "example_config.json"
    custom_config.save(config_file)
    print(f"\n설정 저장: {config_file}")

    # 설정 로드
    loaded_config = AdaptiveSamplingConfig.load(config_file)
    print(f"\n로드된 설정:")
    print(f"  alpha: {loaded_config.alpha}")
    print(f"  beta: {loaded_config.beta}")
    print(f"  normalize: {loaded_config.normalize}")


def example_4_fps_snapping():
    """
    예시 4: FPS 기반 timestamp snapping
    """
    print("\n" + "="*70)
    print("예시 4: FPS 기반 Timestamp Snapping")
    print("="*70)

    poses = [
        CameraPose(i+1, 1, 0, 0, 0, i*0.5, 0, 0, 1, f"f{i}.png", i*0.033)
        for i in range(10)
    ]

    analyzer = TrajectoryAnalyzer(alpha=1.0, beta=0.0, normalize=False)
    segments = analyzer.analyze_trajectory(poses)

    # 30fps로 제한
    sampler = AdaptiveSampler(video_fps=30.0)
    timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=5)

    print(f"\n30fps로 스냅된 timestamp:")
    for i, ts in enumerate(timestamps):
        frame_num = ts * 30.0
        print(f"  Frame {i}: {ts:.6f}s (frame #{frame_num:.1f})")


def main():
    """모든 예시 실행"""
    print("\n" + "="*70)
    print("3DGS ADAPTIVE FRAME SAMPLING - 사용 예시")
    print("="*70)

    example_1_basic_analysis()
    example_2_adaptive_sampling()
    example_3_config()
    example_4_fps_snapping()

    print("\n" + "="*70)
    print("모든 예시 완료!")
    print("="*70)
    print("\n자세한 사용법은 README_ADAPTIVE_SAMPLING.md를 참고하세요.")


if __name__ == "__main__":
    main()
