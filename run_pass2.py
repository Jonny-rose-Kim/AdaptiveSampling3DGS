#!/usr/bin/env python3
"""
Pass 2: Adaptive Sampling으로 프레임 재추출

사용법:
    python run_pass2.py <작업_디렉토리> <목표_프레임수>

예시:
    python run_pass2.py ./museum_output 2000
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from adaptive_sampling.pipeline import PipelineOrchestrator


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    workspace_dir = sys.argv[1]
    desired_frames = int(sys.argv[2])

    print("\n" + "="*70)
    print("Pass 2: Adaptive Sampling")
    print("="*70)

    # 설정 로드
    config_file = Path(workspace_dir) / "pass1_config.json"
    if not config_file.exists():
        print(f"\n❌ Pass 1 설정 파일이 없습니다: {config_file}")
        print(f"먼저 run_pass1.py를 실행하세요.")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = json.load(f)

    print(f"\n로드된 설정:")
    print(f"  비디오: {config['video_path']}")
    print(f"  Pass 1 프레임: {config['extracted_frames']}개")
    print(f"  목표 프레임: {desired_frames}개")
    print(f"  감소율: {(1 - desired_frames/config['extracted_frames'])*100:.1f}%")

    # COLMAP 출력 확인
    colmap_dir = Path(workspace_dir) / "pass1" / "sparse" / "0"
    images_txt = colmap_dir / "images.txt"

    if not images_txt.exists():
        print(f"\n❌ COLMAP 출력이 없습니다: {images_txt}")
        print(f"\nCOLMAP을 먼저 실행해야 합니다:")
        print(f"  python convert.py -s {workspace_dir}/pass1")
        print(f"\n또는 현재 COLMAP이 실행 중인지 확인:")
        print(f"  nvidia-smi")
        print(f"  ps aux | grep colmap")
        sys.exit(1)

    print(f"\n✅ COLMAP 출력 확인: {images_txt}")

    # Pipeline 실행
    pipeline = PipelineOrchestrator(
        video_path=config['video_path'],
        workspace_dir=workspace_dir,
        alpha=0.8,
        beta=0.2,
        normalize=True
    )

    # Trajectory 분석 및 Adaptive Sampling
    # - COLMAP 파싱: extraction_fps 사용 (frame_000001.png → 0.5초)
    # - Adaptive Sampler: 원본 비디오 fps 사용 (timestamp → 정확한 프레임)
    print(f"\n[1/2] Trajectory 분석 및 Adaptive Sampling...")
    print(f"  COLMAP 파싱 fps: {config['extraction_fps']:.2f} (Pass1 추출 fps)")
    print(f"  Adaptive Sampler fps: {config['video_info']['fps']:.2f} (원본 비디오 fps)")

    timestamps, segments, stats = pipeline.analyze_trajectory(
        colmap_dir,
        desired_frame_count=desired_frames,
        extraction_fps=config['extraction_fps']
    )

    print(f"\n분석 결과:")
    print(f"  파싱된 카메라: {len(segments) + 1}개")
    print(f"  총 세그먼트: {stats['num_segments']}")
    print(f"  Translation 평균: {stats['translation']['mean']:.3f}m")
    print(f"  Rotation 평균: {stats['rotation']['mean_degrees']:.2f}°")
    print(f"  계산된 timestamp: {len(timestamps)}개")

    # Pass 2 재추출
    print(f"\n[2/2] Adaptive 프레임 재추출...")
    pass2_files = pipeline.run_pass2(timestamps)

    # 결과 요약
    print(f"\n" + "="*70)
    print("✅ Pass 2 완료!")
    print("="*70)

    print(f"\n결과 요약:")
    print(f"  Pass 1 프레임: {config['extracted_frames']}개")
    print(f"  Pass 2 프레임: {len(pass2_files)}개")
    print(f"  감소율: {(1 - len(pass2_files)/config['extracted_frames'])*100:.1f}%")

    print(f"\n출력 위치:")
    print(f"  Pass 1: {workspace_dir}/pass1/input/")
    print(f"  Pass 2: {workspace_dir}/pass2/images/")
    print(f"  Timestamps: {workspace_dir}/adaptive_timestamps.json")
    print(f"  결과: {workspace_dir}/pipeline_result.json")

    # 다음 단계
    print(f"\n" + "="*70)
    print("다음 단계: 3DGS 학습")
    print("="*70)
    print(f"\nBaseline (Pass 1):")
    print(f"  python train.py -s {workspace_dir}/pass1")

    print(f"\nAdaptive (Pass 2):")
    print(f"  python train.py -s {workspace_dir}/pass2")

    print(f"\n품질 비교:")
    print(f"  python render.py -m <모델_경로>")
    print(f"  python metrics.py -m <모델_경로>")


if __name__ == "__main__":
    main()
