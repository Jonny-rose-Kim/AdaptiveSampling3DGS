"""
3DGS Adaptive Frame Sampling Package

COLMAP 카메라 pose 기반 동적 프레임 추출 시스템

모듈:
- colmap_parser: COLMAP 출력 파싱
- trajectory_analyzer: 카메라 궤적 분석 (geometry 기반)
- adaptive_sampler: geometry 기반 adaptive sampling
- frame_extractor: ffmpeg 기반 프레임 추출

Hybrid 모듈 (SfM Quality 기반):
- sfm_quality_analyzer: SfM 기여도 분석
- hybrid_gap_analyzer: Hybrid Gap Priority 계산
- frame_replacer: 프레임 교체 로직
- hybrid_pipeline: 통합 Hybrid 파이프라인
"""

__version__ = "0.2.0"

# Core modules
from .colmap_parser import COLMAPParser, CameraPose
from .trajectory_analyzer import TrajectoryAnalyzer, TrajectorySegment
from .adaptive_sampler import AdaptiveSampler
from .frame_extractor import FrameExtractor

# Hybrid modules
from .sfm_quality_analyzer import SfMQualityAnalyzer, ImageContribution, FeatureTrackContinuity
from .hybrid_gap_analyzer import HybridGapAnalyzer, HybridGap
from .frame_replacer import FrameReplacer, FrameReplacement
from .hybrid_pipeline import HybridAdaptivePipeline

__all__ = [
    # Core
    'COLMAPParser',
    'CameraPose',
    'TrajectoryAnalyzer',
    'TrajectorySegment',
    'AdaptiveSampler',
    'FrameExtractor',
    # Hybrid
    'SfMQualityAnalyzer',
    'ImageContribution',
    'FeatureTrackContinuity',
    'HybridGapAnalyzer',
    'HybridGap',
    'FrameReplacer',
    'FrameReplacement',
    'HybridAdaptivePipeline',
]
