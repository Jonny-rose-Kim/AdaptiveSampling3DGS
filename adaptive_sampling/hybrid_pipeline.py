"""
Hybrid Adaptive Sampling Pipeline

SfM 품질 기반 Hybrid 접근법의 전체 파이프라인

핵심 아이디어:
1. Pass 1의 COLMAP 결과에서 각 이미지의 SfM 기여도 측정
2. Geometry + Feature Track 연속성을 결합한 Hybrid Gap Priority 계산
3. 기여도 낮은 프레임을 높은 priority gap의 새 timestamp로 교체
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import shutil

try:
    from .colmap_parser import COLMAPParser
    from .trajectory_analyzer import TrajectoryAnalyzer
    from .sfm_quality_analyzer import SfMQualityAnalyzer
    from .hybrid_gap_analyzer import HybridGapAnalyzer
    from .frame_replacer import FrameReplacer
    from .frame_extractor import FrameExtractor
except ImportError:
    from colmap_parser import COLMAPParser
    from trajectory_analyzer import TrajectoryAnalyzer
    from sfm_quality_analyzer import SfMQualityAnalyzer
    from hybrid_gap_analyzer import HybridGapAnalyzer
    from frame_replacer import FrameReplacer
    from frame_extractor import FrameExtractor

logger = logging.getLogger(__name__)


class HybridAdaptivePipeline:
    """SfM Quality 기반 Hybrid Adaptive Sampling 파이프라인"""

    def __init__(
        self,
        video_path: str,
        pass1_dir: str,
        output_dir: str,
        video_fps: float = 30.0,
        extraction_fps: float = 2.0,
        replacement_ratio: float = 0.2,
        geometry_weight: float = 0.5,
        continuity_weight: float = 0.5,
        min_features_threshold: int = 50,
        protect_edge_frames: int = 5
    ):
        """
        Args:
            video_path: 원본 비디오 경로
            pass1_dir: Pass 1 결과 디렉토리 (sparse/0/ 포함)
            output_dir: Pass 2 출력 디렉토리
            video_fps: 원본 비디오 fps
            extraction_fps: Pass 1 프레임 추출 fps
            replacement_ratio: 교체할 프레임 비율 (0.2 = 하위 20%)
            geometry_weight: Hybrid score에서 geometry 가중치
            continuity_weight: Hybrid score에서 continuity 가중치
            min_features_threshold: Textureless 판단 임계값
            protect_edge_frames: 보호할 시작/끝 프레임 수
        """
        self.video_path = Path(video_path)
        self.pass1_dir = Path(pass1_dir)
        self.output_dir = Path(output_dir)
        self.video_fps = video_fps
        self.extraction_fps = extraction_fps
        self.replacement_ratio = replacement_ratio
        self.geometry_weight = geometry_weight
        self.continuity_weight = continuity_weight
        self.min_features_threshold = min_features_threshold
        self.protect_edge_frames = protect_edge_frames

        # COLMAP 디렉토리 확인
        self.colmap_dir = self.pass1_dir / "sparse" / "0"
        if not self.colmap_dir.exists():
            raise FileNotFoundError(f"COLMAP directory not found: {self.colmap_dir}")

        # 분석기 초기화
        self.sfm_analyzer = SfMQualityAnalyzer(
            str(self.colmap_dir),
            extraction_fps=extraction_fps
        )
        self.trajectory_analyzer = TrajectoryAnalyzer(
            alpha=0.7, beta=0.3, normalize=True
        )
        self.gap_analyzer = HybridGapAnalyzer(
            geometry_weight=geometry_weight,
            continuity_weight=continuity_weight
        )
        self.frame_replacer = FrameReplacer(video_fps=video_fps)

        # 비디오 extractor
        self.frame_extractor = FrameExtractor(str(self.video_path))

    def run(self, extract_frames: bool = True) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            extract_frames: True면 새 프레임 추출, False면 분석만 수행

        Returns:
            결과 딕셔너리
        """
        logger.info("=" * 60)
        logger.info("HYBRID ADAPTIVE SAMPLING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Video: {self.video_path}")
        logger.info(f"Pass 1: {self.pass1_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Parameters: replacement_ratio={self.replacement_ratio}, "
                   f"geo_weight={self.geometry_weight}, cont_weight={self.continuity_weight}")

        # Phase 1: SfM 품질 분석
        logger.info("\n[Phase 1/7] Analyzing SfM Quality...")
        contributions = self.sfm_analyzer.parse_images_with_observations()
        continuities = self.sfm_analyzer.compute_feature_track_continuity(contributions)

        contrib_stats = self.sfm_analyzer.get_statistics(contributions)
        cont_stats = self.sfm_analyzer.get_continuity_statistics(continuities)

        logger.info(f"  Images analyzed: {contrib_stats['num_images']}")
        logger.info(f"  Contribution: mean={contrib_stats['contribution']['mean']:.1f}, "
                   f"range=[{contrib_stats['contribution']['min']:.0f}, "
                   f"{contrib_stats['contribution']['max']:.0f}]")
        logger.info(f"  Continuity: mean={cont_stats['continuity_score']['mean']:.3f}, "
                   f"shared_features mean={cont_stats['shared_points']['mean']:.1f}")

        # Phase 2: Geometry 분석 (기존 trajectory analyzer 활용)
        logger.info("\n[Phase 2/7] Analyzing Camera Trajectory...")
        colmap_parser = COLMAPParser(str(self.colmap_dir))
        poses = colmap_parser.parse_and_extract(fps=self.extraction_fps)
        segments = self.trajectory_analyzer.analyze_trajectory(poses)

        traj_stats = self.trajectory_analyzer.get_statistics(segments)
        logger.info(f"  Segments: {traj_stats['num_segments']}")
        logger.info(f"  Translation: mean={traj_stats['translation']['mean']:.3f}m, "
                   f"max={traj_stats['translation']['max']:.3f}m")
        logger.info(f"  Rotation: mean={traj_stats['rotation']['mean_degrees']:.2f}°, "
                   f"max={traj_stats['rotation']['max_degrees']:.2f}°")

        # Phase 3: Hybrid Gap 분석
        logger.info("\n[Phase 3/7] Computing Hybrid Gap Priority...")
        hybrid_gaps = self.gap_analyzer.compute_hybrid_gaps(segments, continuities)

        # Textureless 구간 필터링
        filtered_gaps = self.gap_analyzer.filter_textureless_gaps(
            hybrid_gaps,
            self.min_features_threshold
        )

        gap_stats = self.gap_analyzer.get_statistics(filtered_gaps)
        logger.info(f"  Total gaps: {len(hybrid_gaps)}")
        logger.info(f"  After textureless filtering: {len(filtered_gaps)}")
        logger.info(f"  Priority range: [{gap_stats['priority']['min']:.3f}, "
                   f"{gap_stats['priority']['max']:.3f}]")

        # Phase 4: 기여도 낮은 이미지 선택
        logger.info("\n[Phase 4/7] Selecting Low Contribution Images...")
        low_contrib_images = self.sfm_analyzer.select_low_contribution_images(
            contributions,
            bottom_ratio=self.replacement_ratio,
            protect_edge_frames=self.protect_edge_frames
        )
        logger.info(f"  Selected for replacement: {len(low_contrib_images)}")
        if low_contrib_images:
            logger.info(f"  Contribution range: [{low_contrib_images[0].contribution_score:.0f}, "
                       f"{low_contrib_images[-1].contribution_score:.0f}]")

        # Phase 5: High priority gaps 선택
        logger.info("\n[Phase 5/7] Selecting High Priority Gaps...")
        high_priority_gaps = self.gap_analyzer.select_high_priority_gaps(
            filtered_gaps,
            top_k=len(low_contrib_images)
        )
        logger.info(f"  High priority gaps selected: {len(high_priority_gaps)}")

        # Phase 6: 프레임 교체 계획
        logger.info("\n[Phase 6/7] Computing Frame Replacements...")
        original_timestamps = sorted([c.timestamp for c in contributions.values()])
        existing_timestamps = set(original_timestamps)

        replacements = self.frame_replacer.compute_replacements(
            low_contrib_images,
            high_priority_gaps,
            existing_timestamps
        )
        logger.info(f"  Planned replacements: {len(replacements)}")

        # 교체 보고서 출력
        self.frame_replacer.print_replacement_report(replacements, show_all=False)

        # Phase 7: 최종 timestamp 리스트 생성 및 프레임 추출
        logger.info("\n[Phase 7/7] Generating Final Timestamps...")
        final_timestamps = self.frame_replacer.generate_final_timestamps(
            original_timestamps,
            replacements
        )
        logger.info(f"  Original frames: {len(original_timestamps)}")
        logger.info(f"  Final frames: {len(final_timestamps)}")
        logger.info(f"  Replaced: {len(replacements)}")

        # 프레임 추출
        if extract_frames:
            self._extract_and_organize_frames(
                original_timestamps,
                final_timestamps,
                replacements
            )

        # 결과 저장
        result = self._generate_result(
            contrib_stats, cont_stats, traj_stats, gap_stats,
            replacements, original_timestamps, final_timestamps
        )

        # JSON 저장
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.output_dir / "hybrid_sampling_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nResults saved to: {result_path}")

        return result

    def _extract_and_organize_frames(
        self,
        original_timestamps: List[float],
        final_timestamps: List[float],
        replacements
    ) -> None:
        """프레임 추출 및 정리"""
        logger.info("\n[Extracting Frames]")

        # 출력 디렉토리 생성
        input_dir = self.output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # 유지할 프레임 (교체되지 않는 것들)
        remove_timestamps = {r.remove_timestamp for r in replacements}
        keep_timestamps = [ts for ts in original_timestamps if ts not in remove_timestamps]

        # Pass 1의 images 디렉토리에서 유지할 프레임 복사
        pass1_images = self.pass1_dir / "images"
        if not pass1_images.exists():
            pass1_images = self.pass1_dir / "input"

        if pass1_images.exists():
            logger.info(f"  Copying {len(keep_timestamps)} frames from Pass 1...")
            frame_files = sorted(pass1_images.glob("frame_*.png"))

            # timestamp -> file 매핑 생성
            ts_to_file = {}
            for f in frame_files:
                # frame_NNNNNN.png 또는 frame_NNNNNN_TTT.ttt.png 형식
                import re
                match = re.search(r'frame_(\d+)', f.name)
                if match:
                    frame_num = int(match.group(1))
                    ts = frame_num / self.extraction_fps
                    ts_to_file[round(ts, 3)] = f

            # 유지할 프레임 복사
            copied_count = 0
            for ts in keep_timestamps:
                ts_key = round(ts, 3)
                if ts_key in ts_to_file:
                    src = ts_to_file[ts_key]
                    # 새 파일명: frame_NNNNNN.png (순차 번호)
                    dst = input_dir / src.name
                    shutil.copy2(src, dst)
                    copied_count += 1

            logger.info(f"  Copied {copied_count} existing frames")

        # 새로운 프레임 추출
        new_timestamps = [r.new_timestamp for r in replacements]
        if new_timestamps:
            logger.info(f"  Extracting {len(new_timestamps)} new frames from video...")

            # 원본 비디오에서 새 프레임 추출
            video_info = self.frame_extractor.get_video_info()
            logger.info(f"  Video: {video_info['duration']:.1f}s @ {video_info['fps']:.2f}fps")

            # 임시 디렉토리에 추출
            temp_dir = self.output_dir / "temp_frames"
            temp_dir.mkdir(parents=True, exist_ok=True)

            extracted = self.frame_extractor.extract_frames_by_timestamps(
                new_timestamps,
                str(temp_dir),
                quality=2,
                accurate_seek=True
            )

            # 추출된 프레임을 input 디렉토리로 이동
            for f in Path(temp_dir).glob("frame_*.png"):
                shutil.move(str(f), str(input_dir / f.name))

            # 임시 디렉토리 삭제
            shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info(f"  Extracted {len(extracted)} new frames")

        # 최종 파일 목록 정리 및 재명명
        self._rename_frames_sequentially(input_dir)

    def _rename_frames_sequentially(self, input_dir: Path) -> None:
        """프레임 파일을 순차적으로 재명명"""
        import re

        files = list(input_dir.glob("frame_*.png"))

        # timestamp 정보 추출 및 정렬
        file_with_ts = []
        for f in files:
            # frame_NNNNNN_TTT.ttt.png 형식에서 timestamp 추출 (새로 추출한 프레임)
            match = re.search(r'frame_\d+_(\d+\.\d+)\.png', f.name)
            if match:
                ts = float(match.group(1))
            else:
                # frame_NNNNNN.png 형식 (기존 프레임)
                match = re.search(r'frame_(\d+)\.png', f.name)
                if match:
                    frame_num = int(match.group(1))
                    ts = frame_num / self.extraction_fps
                else:
                    ts = 0.0
            file_with_ts.append((f, ts))

        # timestamp 순으로 정렬
        file_with_ts.sort(key=lambda x: x[1])

        logger.info(f"  Processing {len(file_with_ts)} frames for renaming...")

        # Step 1: 모든 파일을 임시 이름으로 이동 (충돌 방지)
        temp_mappings = []
        for i, (old_path, ts) in enumerate(file_with_ts):
            temp_path = input_dir / f"_temp_{i:06d}_{ts:.3f}.png"
            shutil.move(str(old_path), str(temp_path))
            temp_mappings.append((temp_path, i))

        # Step 2: 임시 파일을 순차적 이름으로 변경
        for temp_path, i in temp_mappings:
            new_name = f"frame_{i+1:06d}.png"
            new_path = input_dir / new_name
            shutil.move(str(temp_path), str(new_path))

        logger.info(f"  Renamed {len(file_with_ts)} frames sequentially")

    def _generate_result(
        self,
        contrib_stats, cont_stats, traj_stats, gap_stats,
        replacements, original_timestamps, final_timestamps
    ) -> Dict:
        """결과 딕셔너리 생성"""
        replacement_summary = self.frame_replacer.get_replacement_summary(replacements)

        return {
            'config': {
                'video_path': str(self.video_path),
                'pass1_dir': str(self.pass1_dir),
                'output_dir': str(self.output_dir),
                'video_fps': self.video_fps,
                'extraction_fps': self.extraction_fps,
                'replacement_ratio': self.replacement_ratio,
                'geometry_weight': self.geometry_weight,
                'continuity_weight': self.continuity_weight,
                'min_features_threshold': self.min_features_threshold,
                'protect_edge_frames': self.protect_edge_frames,
            },
            'statistics': {
                'contribution': contrib_stats,
                'continuity': cont_stats,
                'trajectory': traj_stats,
                'gap': gap_stats,
            },
            'replacements': replacement_summary,
            'final_timestamps': final_timestamps,
            'frame_count': {
                'original': len(original_timestamps),
                'final': len(final_timestamps),
                'replaced': len(replacements),
                'kept': len(original_timestamps) - len(replacements),
            }
        }


def main():
    """CLI 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid Adaptive Sampling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_pipeline.py \\
      --video /path/to/video.mp4 \\
      --pass1 /path/to/pass1 \\
      --output /path/to/pass2_hybrid \\
      --replacement-ratio 0.2

  python hybrid_pipeline.py \\
      --video ./data/Museum.mp4 \\
      --pass1 ./data/Museum_cut_exp/pass1 \\
      --output ./data/Museum_cut_exp/pass2_hybrid \\
      --video-fps 30 --extraction-fps 2 \\
      --geometry-weight 0.5 --continuity-weight 0.5
        """
    )

    parser.add_argument("--video", "-v", required=True,
                       help="Path to original video file")
    parser.add_argument("--pass1", "-p", required=True,
                       help="Path to Pass 1 result directory")
    parser.add_argument("--output", "-o", required=True,
                       help="Path to output directory for Pass 2")
    parser.add_argument("--video-fps", type=float, default=30.0,
                       help="Original video FPS (default: 30.0)")
    parser.add_argument("--extraction-fps", type=float, default=2.0,
                       help="Pass 1 extraction FPS (default: 2.0)")
    parser.add_argument("--replacement-ratio", type=float, default=0.2,
                       help="Ratio of frames to replace (default: 0.2)")
    parser.add_argument("--geometry-weight", type=float, default=0.5,
                       help="Weight for geometry score (default: 0.5)")
    parser.add_argument("--continuity-weight", type=float, default=0.5,
                       help="Weight for continuity score (default: 0.5)")
    parser.add_argument("--min-features", type=int, default=50,
                       help="Minimum shared features for gap (default: 50)")
    parser.add_argument("--protect-edges", type=int, default=5,
                       help="Number of edge frames to protect (default: 5)")
    parser.add_argument("--no-extract", action="store_true",
                       help="Skip frame extraction (analysis only)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 파이프라인 실행
    pipeline = HybridAdaptivePipeline(
        video_path=args.video,
        pass1_dir=args.pass1,
        output_dir=args.output,
        video_fps=args.video_fps,
        extraction_fps=args.extraction_fps,
        replacement_ratio=args.replacement_ratio,
        geometry_weight=args.geometry_weight,
        continuity_weight=args.continuity_weight,
        min_features_threshold=args.min_features,
        protect_edge_frames=args.protect_edges
    )

    result = pipeline.run(extract_frames=not args.no_extract)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original frames: {result['frame_count']['original']}")
    print(f"Replaced frames: {result['frame_count']['replaced']}")
    print(f"Final frames: {result['frame_count']['final']}")
    print(f"\nNext steps:")
    print(f"  1. Run COLMAP on the new frames:")
    print(f"     python convert.py -s {args.output}")
    print(f"  2. Train 3DGS:")
    print(f"     python train.py -s {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
