#!/usr/bin/env python3
"""
Hybrid Adaptive Sampling - 실행 스크립트

SfM Quality 기반 Hybrid 접근법으로 프레임을 교체합니다.

기존 방식과의 차이:
- 기존: 같은 프레임 풀에서 geometry 기준으로 재선택 (새로운 정보 없음)
- Hybrid: 기여도 낮은 프레임을 새로운 timestamp 프레임으로 교체 (새로운 정보 추가)

사용법:
    python run_hybrid.py <video_path> <pass1_dir> <output_dir> [options]

예시:
    python run_hybrid.py \\
        ./data/Museum.mp4 \\
        ./data/Museum_cut_exp/pass1 \\
        ./data/Museum_cut_exp/pass2_hybrid \\
        --replacement-ratio 0.2

    # 분석만 수행 (프레임 추출 없이)
    python run_hybrid.py \\
        ./data/Museum.mp4 \\
        ./data/Museum_cut_exp/pass1 \\
        ./data/Museum_cut_exp/pass2_hybrid \\
        --no-extract
"""

import sys
import logging
from pathlib import Path

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from adaptive_sampling.hybrid_pipeline import HybridAdaptivePipeline


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid Adaptive Sampling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
파이프라인 단계:
  1. SfM 품질 분석: 각 이미지의 기여도 (valid_observations) 측정
  2. 카메라 궤적 분석: geometry 기반 세그먼트 점수 계산
  3. Hybrid Gap Priority: geometry + feature track 연속성 결합
  4. 프레임 교체: 기여도 하위 K장 → 높은 priority gap의 새 프레임으로 교체
  5. COLMAP 재실행 필요

예시:
  # Museum 데이터셋에서 하위 20%% 프레임 교체
  python run_hybrid.py \\
      ./data/Museum.mp4 \\
      ./data/Museum_cut_exp/pass1 \\
      ./data/Museum_cut_exp/pass2_hybrid

  # 하위 10%% 프레임만 교체 (보수적)
  python run_hybrid.py \\
      ./data/Museum.mp4 \\
      ./data/Museum_cut_exp/pass1 \\
      ./data/Museum_cut_exp/pass2_hybrid_10 \\
      --replacement-ratio 0.1

실행 후:
  # COLMAP 실행
  python convert.py -s <output_dir>

  # 3DGS 학습
  python train.py -s <output_dir>
        """
    )

    parser.add_argument("video", type=str,
                       help="원본 비디오 파일 경로")
    parser.add_argument("pass1_dir", type=str,
                       help="Pass 1 결과 디렉토리 (sparse/0/ 포함)")
    parser.add_argument("output_dir", type=str,
                       help="Pass 2 출력 디렉토리")

    parser.add_argument("--video-fps", type=float, default=None,
                       help="원본 비디오 FPS (자동 감지)")
    parser.add_argument("--extraction-fps", type=float, default=2.0,
                       help="Pass 1 프레임 추출 FPS (default: 2.0)")
    parser.add_argument("--replacement-ratio", "-r", type=float, default=0.2,
                       help="교체할 프레임 비율 (default: 0.2 = 하위 20%%)")
    parser.add_argument("--geometry-weight", type=float, default=0.5,
                       help="Geometry 가중치 (default: 0.5)")
    parser.add_argument("--continuity-weight", type=float, default=0.5,
                       help="Continuity 가중치 (default: 0.5)")
    parser.add_argument("--min-features", type=int, default=50,
                       help="Textureless 판단 최소 feature 수 (default: 50)")
    parser.add_argument("--protect-edges", type=int, default=5,
                       help="보호할 시작/끝 프레임 수 (default: 5)")
    parser.add_argument("--no-extract", action="store_true",
                       help="프레임 추출 없이 분석만 수행")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="상세 로그 출력")

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 경로 확인
    video_path = Path(args.video)
    pass1_dir = Path(args.pass1_dir)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    if not pass1_dir.exists():
        print(f"Error: Pass 1 directory not found: {pass1_dir}")
        sys.exit(1)

    colmap_dir = pass1_dir / "sparse" / "0"
    if not colmap_dir.exists():
        print(f"Error: COLMAP results not found: {colmap_dir}")
        print("Please run COLMAP on Pass 1 first:")
        print(f"  python convert.py -s {pass1_dir}")
        sys.exit(1)

    # 비디오 FPS 자동 감지
    video_fps = args.video_fps
    if video_fps is None:
        from adaptive_sampling.frame_extractor import FrameExtractor
        extractor = FrameExtractor(str(video_path))
        video_info = extractor.get_video_info()
        video_fps = video_info['fps']
        print(f"Auto-detected video FPS: {video_fps:.2f}")

    # 파이프라인 실행
    print("\n" + "=" * 60)
    print("HYBRID ADAPTIVE SAMPLING")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Pass 1: {pass1_dir}")
    print(f"Output: {output_dir}")
    print(f"Replacement ratio: {args.replacement_ratio:.0%}")
    print("=" * 60 + "\n")

    pipeline = HybridAdaptivePipeline(
        video_path=str(video_path),
        pass1_dir=str(pass1_dir),
        output_dir=str(output_dir),
        video_fps=video_fps,
        extraction_fps=args.extraction_fps,
        replacement_ratio=args.replacement_ratio,
        geometry_weight=args.geometry_weight,
        continuity_weight=args.continuity_weight,
        min_features_threshold=args.min_features,
        protect_edge_frames=args.protect_edges
    )

    result = pipeline.run(extract_frames=not args.no_extract)

    # 결과 요약
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Original frames: {result['frame_count']['original']}")
    print(f"Replaced frames: {result['frame_count']['replaced']}")
    print(f"Final frames: {result['frame_count']['final']}")

    if not args.no_extract:
        print(f"\nOutput directory: {output_dir}")
        print(f"  - input/: {result['frame_count']['final']} frames")
        print(f"  - hybrid_sampling_result.json: Analysis results")

        print(f"\n다음 단계:")
        print(f"  1. COLMAP 실행:")
        print(f"     python convert.py -s {output_dir}")
        print(f"  2. 3DGS 학습:")
        print(f"     python train.py -s {output_dir}")
        print(f"  3. (선택) Pass 1과 비교:")
        print(f"     python train.py -s {pass1_dir}")
    else:
        print(f"\n분석 결과: {output_dir}/hybrid_sampling_result.json")
        print("프레임 추출을 위해 --no-extract 옵션 없이 다시 실행하세요.")

    print("=" * 60)


if __name__ == "__main__":
    main()
