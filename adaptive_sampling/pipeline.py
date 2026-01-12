"""
Pipeline Orchestrator

전체 2-pass adaptive sampling 파이프라인을 조율합니다.
"""

import subprocess
from pathlib import Path
from typing import Optional, List
import logging
import json
import shutil

try:
    from .frame_extractor import FrameExtractor
    from .colmap_parser import COLMAPParser
    from .trajectory_analyzer import TrajectoryAnalyzer
    from .adaptive_sampler import AdaptiveSampler
except ImportError:
    # For direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from adaptive_sampling.frame_extractor import FrameExtractor
    from adaptive_sampling.colmap_parser import COLMAPParser
    from adaptive_sampling.trajectory_analyzer import TrajectoryAnalyzer
    from adaptive_sampling.adaptive_sampler import AdaptiveSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """2-Pass Adaptive Sampling Pipeline"""

    def __init__(
        self,
        video_path: str,
        workspace_dir: str,
        alpha: float = 0.5,
        beta: float = 0.5,
        normalize: bool = True
    ):
        """
        Args:
            video_path: 입력 비디오 경로
            workspace_dir: 작업 디렉토리
            alpha: Translation 가중치
            beta: Rotation 가중치
            normalize: Score 정규화 여부
        """
        self.video_path = Path(video_path)
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.alpha = alpha
        self.beta = beta
        self.normalize = normalize

        # 서브디렉토리
        self.pass1_dir = self.workspace / "pass1"
        self.pass2_dir = self.workspace / "pass2"

        self.extractor = FrameExtractor(str(self.video_path))
        self.video_info = self.extractor.get_video_info()

        logger.info(f"Pipeline initialized")
        logger.info(f"  Video: {self.video_path}")
        logger.info(f"  FPS: {self.video_info['fps']:.2f}")
        logger.info(f"  Duration: {self.video_info['duration']:.2f}s")
        logger.info(f"  Workspace: {self.workspace}")

    def run_pass1(self, fps: Optional[float] = None) -> tuple:
        """
        Pass 1: 초기 프레임 추출 및 COLMAP 실행

        Args:
            fps: 추출 fps (None이면 원본 fps 사용)

        Returns:
            (extracted_files, colmap_dir)
        """
        logger.info("\n" + "="*60)
        logger.info("PASS 1: Initial Extraction & COLMAP")
        logger.info("="*60)

        # 1. 프레임 추출
        frames_dir = self.pass1_dir / "images"
        frames_dir.mkdir(parents=True, exist_ok=True)

        extraction_fps = fps if fps is not None else self.video_info['fps']
        logger.info(f"\n[1/2] Extracting frames at {extraction_fps:.2f} fps...")

        extracted_files, metadata = self.extractor.extract_frames_with_metadata(
            str(frames_dir),
            fps=extraction_fps
        )

        logger.info(f"  Extracted {len(extracted_files)} frames")

        # 2. COLMAP 실행 (실제로는 외부 스크립트/명령어 호출)
        logger.info(f"\n[2/2] Running COLMAP...")
        logger.info(f"  NOTE: This is a placeholder. In production, run COLMAP here.")
        logger.info(f"  Input: {frames_dir}")

        # COLMAP sparse reconstruction 경로
        colmap_dir = self.pass1_dir / "sparse" / "0"

        # 실제 환경에서는 여기서 COLMAP을 실행:
        # self._run_colmap(str(frames_dir), str(colmap_dir.parent))

        logger.info(f"  Expected output: {colmap_dir}/images.txt")

        return extracted_files, colmap_dir

    def analyze_trajectory(
        self,
        colmap_dir: Path,
        desired_frame_count: int,
        extraction_fps: Optional[float] = None
    ) -> tuple:
        """
        Trajectory 분석 및 adaptive timestamp 계산

        Args:
            colmap_dir: COLMAP sparse reconstruction 디렉토리
            desired_frame_count: 원하는 프레임 수
            extraction_fps: Pass1 프레임 추출 fps (COLMAP 파싱용, None이면 video fps 사용)

        Returns:
            (timestamps, segments, stats)
        """
        logger.info("\n" + "="*60)
        logger.info("Trajectory Analysis & Adaptive Sampling")
        logger.info("="*60)

        # 1. COLMAP 파싱
        logger.info(f"\n[1/4] Parsing COLMAP output...")
        parser = COLMAPParser(str(colmap_dir))
        # COLMAP 파싱: extraction_fps 사용 (프레임 파일명 → 실제 비디오 timestamp 변환)
        parsing_fps = extraction_fps if extraction_fps is not None else self.video_info['fps']
        poses = parser.parse_and_extract(fps=parsing_fps)

        logger.info(f"  Parsed {len(poses)} camera poses (using fps={parsing_fps:.2f})")

        # 2. Trajectory 분석
        logger.info(f"\n[2/4] Analyzing camera trajectory...")
        analyzer = TrajectoryAnalyzer(
            alpha=self.alpha,
            beta=self.beta,
            normalize=self.normalize
        )
        segments = analyzer.analyze_trajectory(poses)

        stats = analyzer.get_statistics(segments)
        logger.info(f"  Total segments: {stats['num_segments']}")
        logger.info(f"  Translation: mean={stats['translation']['mean']:.3f}, "
                   f"std={stats['translation']['std']:.3f}")
        logger.info(f"  Rotation: mean={stats['rotation']['mean_degrees']:.2f}°, "
                   f"std={stats['rotation']['std_degrees']:.2f}°")
        logger.info(f"  Total score: {stats['total_score']:.3f}")

        # 3. Sparse 구간 식별
        logger.info(f"\n[3/4] Identifying sparse regions...")
        sparse_regions = analyzer.identify_sparse_regions(segments, threshold_multiplier=2.0)
        logger.info(f"  Found {len(sparse_regions)} sparse region(s)")
        for start_idx, end_idx in sparse_regions:
            t_start = segments[start_idx].start_pose.timestamp
            t_end = segments[end_idx - 1].end_pose.timestamp if end_idx > 0 else segments[-1].end_pose.timestamp
            logger.info(f"    Region: segment {start_idx}-{end_idx-1} "
                       f"(t={t_start:.2f}s - {t_end:.2f}s)")

        # 4. Adaptive sampling
        logger.info(f"\n[4/4] Computing adaptive timestamps...")
        sampler = AdaptiveSampler(video_fps=self.video_info['fps'])
        base_timestamps = sampler.compute_target_timestamps(segments, desired_frame_count)

        logger.info(f"  Computed {len(base_timestamps)} base timestamps")

        # 5. Sparse region handling
        if len(sparse_regions) > 0:
            logger.info(f"\n[5/5] Handling sparse regions...")
            additional_timestamps = sampler.handle_sparse_regions(
                segments,
                sparse_regions,
                densification_factor=2
            )
            logger.info(f"  Generated {len(additional_timestamps)} additional timestamps")

            # 병합 및 중복 제거
            all_timestamps = sorted(set(base_timestamps + additional_timestamps))
            logger.info(f"  Total after merge: {len(all_timestamps)} timestamps")

            # desired_frame_count 초과 시 score 기반 선택
            if len(all_timestamps) > desired_frame_count:
                logger.info(f"  Limiting to {desired_frame_count} timestamps (score-based selection)")
                timestamps = sampler.merge_timestamps_with_limit(
                    base_timestamps,
                    additional_timestamps,
                    segments,
                    desired_frame_count
                )
                removed_count = len(all_timestamps) - len(timestamps)
                logger.info(f"  Removed {removed_count} low-score frames")
                logger.info(f"  Final: {len(timestamps)} timestamps")
            else:
                timestamps = all_timestamps
                logger.info(f"  Using all {len(timestamps)} timestamps")
        else:
            timestamps = base_timestamps
            logger.info(f"  No sparse regions - using base timestamps only")

        logger.info(f"\nFinal: {len(timestamps)} timestamps")
        logger.info(f"  Time range: {timestamps[0]:.2f}s - {timestamps[-1]:.2f}s")

        # Timestamp 저장
        timestamp_file = self.workspace / "adaptive_timestamps.json"
        sparse_region_info = []
        for start_idx, end_idx in sparse_regions:
            sparse_region_info.append({
                'start_segment': start_idx,
                'end_segment': end_idx - 1,
                'start_time': segments[start_idx].start_pose.timestamp,
                'end_time': segments[end_idx - 1].end_pose.timestamp if end_idx > 0 else segments[-1].end_pose.timestamp
            })

        with open(timestamp_file, 'w') as f:
            json.dump({
                'timestamps': timestamps,
                'desired_frame_count': desired_frame_count,
                'actual_frame_count': len(timestamps),
                'base_frame_count': len(base_timestamps),
                'additional_frame_count': len(timestamps) - len(base_timestamps) if len(sparse_regions) > 0 else 0,
                'video_fps': self.video_info['fps'],
                'sparse_regions': sparse_region_info,
                'statistics': stats
            }, f, indent=2)

        logger.info(f"  Saved timestamps to {timestamp_file}")

        return timestamps, segments, stats

    def run_pass2(self, timestamps: List[float]) -> List[str]:
        """
        Pass 2: Adaptive timestamp로 재추출

        Args:
            timestamps: 추출할 timestamp 리스트

        Returns:
            추출된 파일 리스트
        """
        logger.info("\n" + "="*60)
        logger.info("PASS 2: Adaptive Frame Re-extraction")
        logger.info("="*60)

        frames_dir = self.pass2_dir / "images"
        frames_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nExtracting {len(timestamps)} frames by timestamps...")

        extracted_files = self.extractor.extract_frames_by_timestamps(
            timestamps,
            str(frames_dir),
            accurate_seek=True
        )

        logger.info(f"  Successfully extracted {len(extracted_files)} frames")
        logger.info(f"  Output directory: {frames_dir}")

        # 메타데이터 저장
        metadata = {}
        for i, (ts, file_path) in enumerate(zip(timestamps, extracted_files)):
            filename = Path(file_path).name
            metadata[filename] = {
                'index': i,
                'timestamp': ts,
                'path': file_path
            }

        metadata_file = frames_dir / "frame_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return extracted_files

    def run_full_pipeline(
        self,
        desired_frame_count: int,
        pass1_fps: Optional[float] = None
    ) -> dict:
        """
        전체 2-pass 파이프라인 실행

        Args:
            desired_frame_count: 최종 원하는 프레임 수
            pass1_fps: Pass 1 추출 fps (None이면 원본 fps)

        Returns:
            결과 딕셔너리
        """
        logger.info("\n" + "="*70)
        logger.info("3DGS ADAPTIVE FRAME SAMPLING PIPELINE")
        logger.info("="*70)

        # Pass 1
        pass1_files, colmap_dir = self.run_pass1(fps=pass1_fps)

        # Trajectory Analysis (COLMAP이 완료되었다고 가정)
        # 실제 환경에서는 COLMAP 완료 대기 필요
        if not (colmap_dir / "images.txt").exists():
            logger.warning(f"\nCOLMAP output not found: {colmap_dir}/images.txt")
            logger.warning(f"Please run COLMAP manually and then continue with:")
            logger.warning(f"  python -m adaptive_sampling.pipeline analyze {self.workspace} {desired_frame_count}")
            return {
                'pass1_completed': True,
                'pass1_frames': len(pass1_files),
                'pass1_dir': str(self.pass1_dir),
                'colmap_dir': str(colmap_dir),
                'status': 'waiting_for_colmap'
            }

        timestamps, segments, stats = self.analyze_trajectory(colmap_dir, desired_frame_count)

        # Pass 2
        pass2_files = self.run_pass2(timestamps)

        # 최종 결과
        result = {
            'status': 'completed',
            'pass1_frames': len(pass1_files),
            'pass2_frames': len(pass2_files),
            'desired_frames': desired_frame_count,
            'pass1_dir': str(self.pass1_dir),
            'pass2_dir': str(self.pass2_dir),
            'video_info': self.video_info,
            'statistics': stats
        }

        # 결과 저장
        result_file = self.workspace / "pipeline_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED")
        logger.info("="*70)
        logger.info(f"\nResults:")
        logger.info(f"  Pass 1 frames: {len(pass1_files)}")
        logger.info(f"  Pass 2 frames: {len(pass2_files)}")
        logger.info(f"  Output: {self.pass2_dir / 'images'}")
        logger.info(f"  Result file: {result_file}")

        return result

    def _run_colmap(self, images_dir: str, output_dir: str):
        """
        COLMAP 실행 (placeholder)

        실제 구현에서는 COLMAP feature extraction, matching, sparse reconstruction 실행
        """
        # 예시 COLMAP 명령어들:
        # colmap feature_extractor --database_path database.db --image_path images/
        # colmap exhaustive_matcher --database_path database.db
        # colmap mapper --database_path database.db --image_path images/ --output_path sparse/
        pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python pipeline.py <video_path> <workspace_dir> <desired_frame_count> [pass1_fps]")
        print("\nExample:")
        print("  python pipeline.py video.mp4 ./workspace 100 30")
        sys.exit(1)

    video_path = sys.argv[1]
    workspace_dir = sys.argv[2]
    desired_frame_count = int(sys.argv[3])
    pass1_fps = float(sys.argv[4]) if len(sys.argv) > 4 else None

    pipeline = PipelineOrchestrator(
        video_path,
        workspace_dir,
        alpha=0.5,
        beta=0.5,
        normalize=True
    )

    result = pipeline.run_full_pipeline(
        desired_frame_count=desired_frame_count,
        pass1_fps=pass1_fps
    )

    print(f"\nPipeline result:")
    print(json.dumps(result, indent=2))
