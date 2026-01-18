# SfM Quality-Based Hybrid Adaptive Sampling

## 개요

기존 Geometry 기반 adaptive sampling의 한계를 극복하기 위한 **SfM 품질 기반 Hybrid 접근법** 구현 계획서입니다.

### 연구 Contribution

**Contribution 1**: 같은 프레임 수로 더 좋은 품질
> "같은 N장이라도 **어떤 프레임**을 뽑느냐에 따라 SfM/3DGS 품질이 달라진다"

**Contribution 2**: 최소 프레임 추가로 최대 효과  
> "fps 2→3 (1.5배 증가) 전체 추가보다, **선택적 소량 추가**가 더 효율적이다"

---

## 기존 방식의 문제점

### 현재 구현 (Geometry 기반)
```
Pass 1: Uniform fps로 N장 추출 → COLMAP
Pass 2: 카메라 pose 분석 → 균등한 physical spacing으로 프레임 재선택
```

### 문제점

| 문제 | 설명 |
|------|------|
| **같은 프레임 풀** | Pass 2에서 선택하는 프레임이 Pass 1의 부분집합 |
| **새로운 정보 없음** | 기존 프레임을 솎아내는 것이지 추가가 아님 |
| **SfM 품질 무시** | 실제 SfM 기여도와 무관하게 geometry만 고려 |

### 실험 결과 (현재)
```
Original (uniform fps=2):  18.2k points, PSNR 30.81 (30K iter)
Adaptive (α=0.5, β=0.5):   17.88k points, PSNR 30.59 (30K iter)
```
→ SfM points 감소, PSNR 하락

---

## 새로운 접근법: SfM Quality-Based Hybrid Sampling

### 핵심 아이디어

1. **기여도 낮은 이미지 식별**: COLMAP 결과에서 각 이미지의 SfM 기여도 측정
2. **Hybrid Gap Priority**: Geometry + Feature Track 연속성 결합
3. **프레임 교체**: 기여도 낮은 프레임을 높은 priority gap의 새 timestamp로 교체

### 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 초기 추출 및 SfM                                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Uniform fps=2로 N장 추출                                      │
│  2. COLMAP 실행 → images.txt, points3D.txt 생성                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: SfM 품질 분석                                          │
├─────────────────────────────────────────────────────────────────┤
│  3. 각 이미지의 SfM 기여도(valid_observations) 계산               │
│  4. 인접 이미지 간 Feature Track 연속성 계산                       │
│  5. 카메라 간 Geometry 거리 계산 (기존 구현 활용)                   │
│  6. Hybrid Gap Priority Score 계산                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: 프레임 교체                                            │
├─────────────────────────────────────────────────────────────────┤
│  7. 기여도 하위 K장 선택 (교체 대상)                               │
│  8. Gap Priority 상위 K개 구간 선택 (삽입 위치)                    │
│  9. 새 timestamp 계산 (gap 중간 지점)                             │
│  10. 원본 비디오에서 새 프레임 추출                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: 재구성 및 검증                                         │
├─────────────────────────────────────────────────────────────────┤
│  11. 새 프레임 세트로 COLMAP 재실행                               │
│  12. SfM points 수 비교                                          │
│  13. 3DGS 학습 및 PSNR 평가                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 구현 상세

### 1. SfM 기여도 계산 모듈

**파일**: `sfm_quality_analyzer.py` (신규 생성)

```python
"""
SfM Quality Analyzer

COLMAP 출력을 분석하여 각 이미지의 SfM 기여도를 계산합니다.
"""

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ImageContribution:
    """이미지의 SfM 기여도 정보"""
    image_id: int
    image_name: str
    timestamp: float
    
    # 기여도 지표
    total_observations: int      # 전체 2D point 수
    valid_observations: int      # 유효한 3D point 관측 수 (POINT3D_ID != -1)
    
    # 추가 분석용 (선택적)
    observed_point3d_ids: Set[int]  # 관측한 3D point ID 집합
    
    @property
    def contribution_score(self) -> float:
        """기본 기여도 점수 = valid_observations"""
        return float(self.valid_observations)
    
    @property
    def observation_ratio(self) -> float:
        """유효 관측 비율"""
        if self.total_observations == 0:
            return 0.0
        return self.valid_observations / self.total_observations


@dataclass  
class FeatureTrackContinuity:
    """인접 이미지 간 Feature Track 연속성"""
    image_i_id: int
    image_j_id: int
    timestamp_i: float
    timestamp_j: float
    
    shared_point_count: int      # 공유하는 3D point 수
    continuity_score: float      # 정규화된 연속성 점수


class SfMQualityAnalyzer:
    """SfM 품질 분석 클래스"""
    
    def __init__(self, colmap_dir: str):
        """
        Args:
            colmap_dir: COLMAP sparse reconstruction 디렉토리 경로
                       (보통 <dataset>/sparse/0/)
        """
        self.colmap_dir = Path(colmap_dir)
        self.images_file = self.colmap_dir / "images.txt"
        self.points3d_file = self.colmap_dir / "points3D.txt"
        
    def parse_images_with_observations(self) -> Dict[int, ImageContribution]:
        """
        images.txt를 파싱하여 각 이미지의 기여도 정보를 추출
        
        Returns:
            image_id를 key로 하는 ImageContribution 딕셔너리
        """
        contributions = {}
        
        with open(self.images_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # 첫째 줄: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            parts = line.split()
            if len(parts) < 10:
                i += 1
                continue
            
            try:
                image_id = int(parts[0])
                image_name = parts[9]
                
                # 파일명에서 timestamp 추출 (기존 colmap_parser 로직 활용)
                timestamp = self._extract_timestamp(image_name)
                
                # 둘째 줄: POINTS2D[] as (X, Y, POINT3D_ID) ...
                i += 1
                if i >= len(lines):
                    break
                    
                points_line = lines[i].strip()
                point3d_ids = self._parse_point3d_ids(points_line)
                
                # 유효한 관측 계산
                valid_ids = [pid for pid in point3d_ids if pid != -1]
                
                contributions[image_id] = ImageContribution(
                    image_id=image_id,
                    image_name=image_name,
                    timestamp=timestamp,
                    total_observations=len(point3d_ids),
                    valid_observations=len(valid_ids),
                    observed_point3d_ids=set(valid_ids)
                )
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line {i}: {e}")
            
            i += 1
        
        return contributions
    
    def _parse_point3d_ids(self, points_line: str) -> List[int]:
        """
        POINTS2D 라인에서 POINT3D_ID들을 추출
        
        Format: X1 Y1 POINT3D_ID1 X2 Y2 POINT3D_ID2 ...
        """
        if not points_line:
            return []
            
        parts = points_line.split()
        point3d_ids = []
        
        # 3개씩 묶어서 처리 (X, Y, POINT3D_ID)
        for j in range(2, len(parts), 3):
            try:
                point3d_id = int(parts[j])
                point3d_ids.append(point3d_id)
            except (ValueError, IndexError):
                continue
        
        return point3d_ids
    
    def _extract_timestamp(self, filename: str, fps: float = 2.0) -> float:
        """파일명에서 timestamp 추출"""
        import re
        
        # frame_NNNNNN.ext 패턴
        match = re.search(r'frame_(\d+)', filename)
        if match:
            frame_number = int(match.group(1))
            return frame_number / fps
        
        # NNNNNN.ext 패턴
        match = re.search(r'(\d+)\.', filename)
        if match:
            frame_number = int(match.group(1))
            return frame_number / fps
        
        return 0.0
    
    def compute_feature_track_continuity(
        self,
        contributions: Dict[int, ImageContribution]
    ) -> List[FeatureTrackContinuity]:
        """
        인접 이미지 간의 Feature Track 연속성 계산
        
        Args:
            contributions: parse_images_with_observations() 결과
            
        Returns:
            시간순으로 정렬된 FeatureTrackContinuity 리스트
        """
        # timestamp 순으로 정렬
        sorted_images = sorted(
            contributions.values(),
            key=lambda x: x.timestamp
        )
        
        continuities = []
        
        for idx in range(len(sorted_images) - 1):
            img_i = sorted_images[idx]
            img_j = sorted_images[idx + 1]
            
            # 공유하는 3D point 계산
            shared_points = img_i.observed_point3d_ids & img_j.observed_point3d_ids
            shared_count = len(shared_points)
            
            # 연속성 점수 = 공유 point 수 / min(두 이미지의 valid observations)
            min_observations = min(img_i.valid_observations, img_j.valid_observations)
            if min_observations > 0:
                continuity_score = shared_count / min_observations
            else:
                continuity_score = 0.0
            
            continuities.append(FeatureTrackContinuity(
                image_i_id=img_i.image_id,
                image_j_id=img_j.image_id,
                timestamp_i=img_i.timestamp,
                timestamp_j=img_j.timestamp,
                shared_point_count=shared_count,
                continuity_score=continuity_score
            ))
        
        return continuities
    
    def select_low_contribution_images(
        self,
        contributions: Dict[int, ImageContribution],
        bottom_ratio: float = 0.2
    ) -> List[ImageContribution]:
        """
        기여도 하위 K장 선택
        
        Args:
            contributions: 이미지 기여도 딕셔너리
            bottom_ratio: 하위 비율 (0.2 = 하위 20%)
            
        Returns:
            기여도 낮은 이미지 리스트 (기여도 오름차순)
        """
        sorted_images = sorted(
            contributions.values(),
            key=lambda x: x.contribution_score
        )
        
        k = int(len(sorted_images) * bottom_ratio)
        k = max(1, k)  # 최소 1개
        
        return sorted_images[:k]
    
    def get_statistics(
        self,
        contributions: Dict[int, ImageContribution]
    ) -> dict:
        """기여도 통계 정보"""
        scores = [c.contribution_score for c in contributions.values()]
        ratios = [c.observation_ratio for c in contributions.values()]
        
        return {
            'num_images': len(contributions),
            'contribution': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
            },
            'observation_ratio': {
                'mean': np.mean(ratios),
                'std': np.std(ratios),
                'min': np.min(ratios),
                'max': np.max(ratios),
            }
        }
```

---

### 2. Hybrid Gap Priority Calculator

**파일**: `hybrid_gap_analyzer.py` (신규 생성)

```python
"""
Hybrid Gap Analyzer

Geometry와 Feature Track 연속성을 결합한 Gap Priority 계산
"""

from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

from sfm_quality_analyzer import FeatureTrackContinuity
from trajectory_analyzer import TrajectorySegment


@dataclass
class HybridGap:
    """Hybrid 분석 기반 Gap 정보"""
    start_timestamp: float
    end_timestamp: float
    
    # Geometry 정보 (기존 TrajectorySegment에서)
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
        
        Args:
            segments: TrajectoryAnalyzer의 출력
            continuities: SfMQualityAnalyzer의 Feature Track 연속성 출력
            
        Returns:
            HybridGap 리스트 (시간순)
        """
        if len(segments) != len(continuities):
            raise ValueError(
                f"Segment count ({len(segments)}) != Continuity count ({len(continuities)})"
            )
        
        # Geometry 정규화를 위한 통계
        geo_scores = [seg.score for seg in segments]
        geo_mean = np.mean(geo_scores)
        geo_std = np.std(geo_scores) + 1e-6
        
        # Continuity 정규화를 위한 통계
        cont_scores = [c.continuity_score for c in continuities]
        cont_mean = np.mean(cont_scores)
        cont_std = np.std(cont_scores) + 1e-6
        
        hybrid_gaps = []
        
        for seg, cont in zip(segments, continuities):
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
        return [
            gap for gap in gaps 
            if gap.shared_features >= min_features_threshold
        ]
    
    def get_statistics(self, gaps: List[HybridGap]) -> dict:
        """Gap 통계 정보"""
        priorities = [g.gap_priority for g in gaps]
        distances = [g.camera_distance for g in gaps]
        continuities = [g.continuity_score for g in gaps]
        features = [g.shared_features for g in gaps]
        
        return {
            'num_gaps': len(gaps),
            'priority': {
                'mean': np.mean(priorities),
                'std': np.std(priorities),
                'min': np.min(priorities),
                'max': np.max(priorities),
            },
            'camera_distance': {
                'mean': np.mean(distances),
                'std': np.std(distances),
            },
            'continuity': {
                'mean': np.mean(continuities),
                'std': np.std(continuities),
            },
            'shared_features': {
                'mean': np.mean(features),
                'min': np.min(features),
                'max': np.max(features),
            }
        }
```

---

### 3. 프레임 교체 로직

**파일**: `frame_replacer.py` (신규 생성)

```python
"""
Frame Replacer

기여도 낮은 프레임을 높은 priority gap의 새 timestamp로 교체
"""

from typing import List, Tuple, Set
from dataclasses import dataclass
import numpy as np

from sfm_quality_analyzer import ImageContribution
from hybrid_gap_analyzer import HybridGap


@dataclass
class FrameReplacement:
    """프레임 교체 정보"""
    # 제거할 프레임
    remove_timestamp: float
    remove_image_name: str
    remove_contribution: float
    
    # 추가할 프레임
    new_timestamp: float
    target_gap: HybridGap
    
    # 교체 근거
    reason: str


class FrameReplacer:
    """프레임 교체 관리 클래스"""
    
    def __init__(self, video_fps: float = 30.0):
        """
        Args:
            video_fps: 원본 비디오 fps
        """
        self.video_fps = video_fps
    
    def compute_replacements(
        self,
        low_contrib_images: List[ImageContribution],
        high_priority_gaps: List[HybridGap],
        existing_timestamps: Set[float]
    ) -> List[FrameReplacement]:
        """
        프레임 교체 계획 생성
        
        Args:
            low_contrib_images: 기여도 낮은 이미지 리스트
            high_priority_gaps: Priority 높은 gap 리스트
            existing_timestamps: 현재 사용 중인 timestamp 집합
            
        Returns:
            FrameReplacement 리스트
        """
        replacements = []
        used_gaps = set()
        
        for img in low_contrib_images:
            # 아직 사용하지 않은 gap 중 가장 높은 priority 선택
            best_gap = None
            for gap in high_priority_gaps:
                gap_key = (gap.start_timestamp, gap.end_timestamp)
                if gap_key not in used_gaps:
                    best_gap = gap
                    break
            
            if best_gap is None:
                continue
            
            # Gap 중간 지점으로 새 timestamp 결정
            new_ts = self._snap_to_video_frame(best_gap.midpoint_timestamp)
            
            # 이미 존재하는 timestamp면 약간 조정
            while new_ts in existing_timestamps:
                new_ts = self._snap_to_video_frame(new_ts + 1/self.video_fps)
            
            replacement = FrameReplacement(
                remove_timestamp=img.timestamp,
                remove_image_name=img.image_name,
                remove_contribution=img.contribution_score,
                new_timestamp=new_ts,
                target_gap=best_gap,
                reason=self._generate_reason(img, best_gap)
            )
            
            replacements.append(replacement)
            used_gaps.add((best_gap.start_timestamp, best_gap.end_timestamp))
            existing_timestamps.add(new_ts)
            existing_timestamps.discard(img.timestamp)
        
        return replacements
    
    def _snap_to_video_frame(self, timestamp: float) -> float:
        """비디오 fps에 맞는 가장 가까운 timestamp로 스냅"""
        frame_interval = 1.0 / self.video_fps
        frame_number = round(timestamp / frame_interval)
        return frame_number * frame_interval
    
    def _generate_reason(
        self,
        img: ImageContribution,
        gap: HybridGap
    ) -> str:
        """교체 근거 문자열 생성"""
        return (
            f"Low contribution ({img.contribution_score:.0f} points) → "
            f"High priority gap (geo={gap.geometry_score:.3f}, "
            f"cont={gap.continuity_score:.3f}, "
            f"priority={gap.gap_priority:.3f})"
        )
    
    def generate_final_timestamps(
        self,
        original_timestamps: List[float],
        replacements: List[FrameReplacement]
    ) -> List[float]:
        """
        최종 timestamp 리스트 생성
        
        Args:
            original_timestamps: 원본 timestamp 리스트
            replacements: 교체 정보 리스트
            
        Returns:
            교체가 반영된 최종 timestamp 리스트
        """
        # 제거할 timestamp
        remove_set = {r.remove_timestamp for r in replacements}
        
        # 유지할 timestamp
        keep_timestamps = [ts for ts in original_timestamps if ts not in remove_set]
        
        # 추가할 timestamp
        add_timestamps = [r.new_timestamp for r in replacements]
        
        # 합치고 정렬
        final_timestamps = sorted(set(keep_timestamps + add_timestamps))
        
        return final_timestamps
    
    def print_replacement_report(self, replacements: List[FrameReplacement]) -> None:
        """교체 보고서 출력"""
        print("\n" + "="*70)
        print("FRAME REPLACEMENT REPORT")
        print("="*70)
        
        for i, r in enumerate(replacements):
            print(f"\n[{i+1}] {r.remove_image_name}")
            print(f"    Remove: t={r.remove_timestamp:.3f}s (contribution={r.remove_contribution:.0f})")
            print(f"    Add:    t={r.new_timestamp:.3f}s")
            print(f"    Gap:    [{r.target_gap.start_timestamp:.3f}s ~ {r.target_gap.end_timestamp:.3f}s]")
            print(f"    Reason: {r.reason}")
        
        print("\n" + "="*70)
        print(f"Total replacements: {len(replacements)}")
        print("="*70)
```

---

### 4. 통합 파이프라인

**파일**: `hybrid_pipeline.py` (신규 생성)

```python
"""
Hybrid Adaptive Sampling Pipeline

SfM 품질 기반 Hybrid 접근법의 전체 파이프라인
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

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
        output_dir: str,
        colmap_dir: str,
        video_fps: float = 30.0,
        extraction_fps: float = 2.0,
        replacement_ratio: float = 0.2,
        geometry_weight: float = 0.5,
        continuity_weight: float = 0.5,
        min_features_threshold: int = 50
    ):
        """
        Args:
            video_path: 원본 비디오 경로
            output_dir: 출력 디렉토리
            colmap_dir: Pass 1 COLMAP 결과 디렉토리 (sparse/0/)
            video_fps: 원본 비디오 fps
            extraction_fps: 프레임 추출 fps
            replacement_ratio: 교체할 프레임 비율 (0.2 = 하위 20%)
            geometry_weight: Hybrid score에서 geometry 가중치
            continuity_weight: Hybrid score에서 continuity 가중치
            min_features_threshold: Textureless 판단 임계값
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.colmap_dir = Path(colmap_dir)
        self.video_fps = video_fps
        self.extraction_fps = extraction_fps
        self.replacement_ratio = replacement_ratio
        self.geometry_weight = geometry_weight
        self.continuity_weight = continuity_weight
        self.min_features_threshold = min_features_threshold
        
        # 분석기 초기화
        self.sfm_analyzer = SfMQualityAnalyzer(str(colmap_dir))
        self.trajectory_analyzer = TrajectoryAnalyzer(alpha=0.7, beta=0.3, normalize=True)
        self.gap_analyzer = HybridGapAnalyzer(geometry_weight, continuity_weight)
        self.frame_replacer = FrameReplacer(video_fps)
        
    def run(self) -> Dict:
        """
        전체 파이프라인 실행
        
        Returns:
            결과 딕셔너리
        """
        logger.info("="*60)
        logger.info("HYBRID ADAPTIVE SAMPLING PIPELINE")
        logger.info("="*60)
        
        # Phase 1: SfM 품질 분석
        logger.info("\n[Phase 1] Analyzing SfM Quality...")
        contributions = self.sfm_analyzer.parse_images_with_observations()
        continuities = self.sfm_analyzer.compute_feature_track_continuity(contributions)
        
        contrib_stats = self.sfm_analyzer.get_statistics(contributions)
        logger.info(f"  Images analyzed: {contrib_stats['num_images']}")
        logger.info(f"  Contribution mean: {contrib_stats['contribution']['mean']:.1f}")
        logger.info(f"  Contribution std: {contrib_stats['contribution']['std']:.1f}")
        
        # Phase 2: Geometry 분석 (기존 trajectory analyzer 활용)
        logger.info("\n[Phase 2] Analyzing Camera Trajectory...")
        colmap_parser = COLMAPParser(str(self.colmap_dir))
        poses = colmap_parser.parse_and_extract(fps=self.extraction_fps)
        segments = self.trajectory_analyzer.analyze_trajectory(poses)
        
        traj_stats = self.trajectory_analyzer.get_statistics(segments)
        logger.info(f"  Segments: {traj_stats['num_segments']}")
        logger.info(f"  Translation mean: {traj_stats['translation']['mean']:.3f}m")
        
        # Phase 3: Hybrid Gap 분석
        logger.info("\n[Phase 3] Computing Hybrid Gap Priority...")
        hybrid_gaps = self.gap_analyzer.compute_hybrid_gaps(segments, continuities)
        
        # Textureless 구간 필터링
        filtered_gaps = self.gap_analyzer.filter_textureless_gaps(
            hybrid_gaps, 
            self.min_features_threshold
        )
        logger.info(f"  Total gaps: {len(hybrid_gaps)}")
        logger.info(f"  After textureless filtering: {len(filtered_gaps)}")
        
        gap_stats = self.gap_analyzer.get_statistics(filtered_gaps)
        logger.info(f"  Priority range: [{gap_stats['priority']['min']:.3f}, {gap_stats['priority']['max']:.3f}]")
        
        # Phase 4: 기여도 낮은 이미지 선택
        logger.info("\n[Phase 4] Selecting Low Contribution Images...")
        low_contrib_images = self.sfm_analyzer.select_low_contribution_images(
            contributions,
            bottom_ratio=self.replacement_ratio
        )
        logger.info(f"  Selected for replacement: {len(low_contrib_images)}")
        
        # Phase 5: High priority gaps 선택
        logger.info("\n[Phase 5] Selecting High Priority Gaps...")
        high_priority_gaps = self.gap_analyzer.select_high_priority_gaps(
            filtered_gaps,
            top_k=len(low_contrib_images)
        )
        logger.info(f"  High priority gaps: {len(high_priority_gaps)}")
        
        # Phase 6: 프레임 교체 계획
        logger.info("\n[Phase 6] Computing Frame Replacements...")
        original_timestamps = sorted([c.timestamp for c in contributions.values()])
        existing_timestamps = set(original_timestamps)
        
        replacements = self.frame_replacer.compute_replacements(
            low_contrib_images,
            high_priority_gaps,
            existing_timestamps
        )
        logger.info(f"  Planned replacements: {len(replacements)}")
        
        # 교체 보고서 출력
        self.frame_replacer.print_replacement_report(replacements)
        
        # Phase 7: 최종 timestamp 리스트 생성
        logger.info("\n[Phase 7] Generating Final Timestamps...")
        final_timestamps = self.frame_replacer.generate_final_timestamps(
            original_timestamps,
            replacements
        )
        logger.info(f"  Original frames: {len(original_timestamps)}")
        logger.info(f"  Final frames: {len(final_timestamps)}")
        
        # 결과 저장
        result = {
            'config': {
                'video_path': str(self.video_path),
                'colmap_dir': str(self.colmap_dir),
                'replacement_ratio': self.replacement_ratio,
                'geometry_weight': self.geometry_weight,
                'continuity_weight': self.continuity_weight,
                'min_features_threshold': self.min_features_threshold,
            },
            'statistics': {
                'contribution': contrib_stats,
                'trajectory': traj_stats,
                'gap': gap_stats,
            },
            'replacements': [
                {
                    'remove_timestamp': r.remove_timestamp,
                    'remove_image_name': r.remove_image_name,
                    'remove_contribution': r.remove_contribution,
                    'new_timestamp': r.new_timestamp,
                    'reason': r.reason,
                }
                for r in replacements
            ],
            'final_timestamps': final_timestamps,
            'frame_count': {
                'original': len(original_timestamps),
                'final': len(final_timestamps),
                'replaced': len(replacements),
            }
        }
        
        # JSON 저장
        result_path = self.output_dir / "hybrid_sampling_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nResults saved to: {result_path}")
        
        return result


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 사용 예시
    pipeline = HybridAdaptivePipeline(
        video_path="/path/to/video.mp4",
        output_dir="/path/to/output",
        colmap_dir="/path/to/pass1/sparse/0",
        video_fps=30.0,
        extraction_fps=2.0,
        replacement_ratio=0.2,
        geometry_weight=0.5,
        continuity_weight=0.5,
    )
    
    result = pipeline.run()
    print(f"\nFinal frame count: {result['frame_count']['final']}")
```

---

## 파일 구조

```
adaptive_sampling/
├── __init__.py
├── colmap_parser.py          # 기존 (수정 없음)
├── trajectory_analyzer.py     # 기존 (수정 없음)
├── adaptive_sampler.py        # 기존 (수정 없음)
├── frame_extractor.py         # 기존 (수정 없음)
├── pipeline.py                # 기존 geometry 기반 파이프라인
│
├── sfm_quality_analyzer.py    # 🆕 SfM 기여도 분석
├── hybrid_gap_analyzer.py     # 🆕 Hybrid Gap Priority 계산
├── frame_replacer.py          # 🆕 프레임 교체 로직
└── hybrid_pipeline.py         # 🆕 통합 Hybrid 파이프라인
```

---

## 실행 방법

### Step 1: Pass 1 실행 (기존 방식)
```bash
# Uniform fps=2로 프레임 추출 및 COLMAP 실행
python pipeline.py \
    --video /path/to/video.mp4 \
    --output ./data/experiment/pass1 \
    --fps 2.0
```

### Step 2: Hybrid Pass 2 실행 (새 방식)
```bash
# SfM 품질 기반 프레임 교체
python hybrid_pipeline.py \
    --video /path/to/video.mp4 \
    --colmap ./data/experiment/pass1/sparse/0 \
    --output ./data/experiment/pass2_hybrid \
    --replacement-ratio 0.2 \
    --geometry-weight 0.5 \
    --continuity-weight 0.5
```

### Step 3: 결과 비교
```bash
# Pass 1 vs Pass 2 SfM points 비교
echo "Pass 1 SfM points:"
grep "Number of 3D points" ./data/experiment/pass1/colmap.log

echo "Pass 2 (Hybrid) SfM points:"
grep "Number of 3D points" ./data/experiment/pass2_hybrid/colmap.log
```

---

## 예상 결과

### 시나리오 A: 정상적인 영상

```
Pass 1 (Uniform):
  - 240 frames
  - 18,200 SfM points
  - PSNR: 30.81

Pass 2 (Hybrid):
  - 240 frames (동일)
  - 19,500+ SfM points (증가 예상)
  - PSNR: 31.xx (개선 예상)

이유:
  - 기여도 낮은 프레임 → 높은 priority gap으로 이동
  - Feature matching 성공률 증가
  - SfM point 품질 향상
```

### 시나리오 B: Textureless 영역이 많은 영상

```
기존 Feature Track만 사용 시:
  - Textureless 구간에 프레임 추가 → 효과 없음
  - 오히려 다른 구간 프레임 낭비

Hybrid 사용 시:
  - Textureless 구간 자동 필터링 (min_features_threshold)
  - 실제 개선 가능한 구간에만 프레임 배치
```

---

## 하이퍼파라미터 가이드

| 파라미터 | 기본값 | 설명 | 조정 가이드 |
|----------|--------|------|-------------|
| `replacement_ratio` | 0.2 | 교체할 프레임 비율 | 0.1~0.3 권장 |
| `geometry_weight` | 0.5 | Geometry 가중치 | 빠른 움직임 많으면 ↑ |
| `continuity_weight` | 0.5 | Continuity 가중치 | Feature 끊김 많으면 ↑ |
| `min_features_threshold` | 50 | Textureless 임계값 | 데이터셋에 따라 조정 |

---

## Contribution 2를 위한 확장 (추후 구현)

### 선택적 프레임 추가

```python
def selective_frame_addition(
    base_timestamps: List[float],
    gaps: List[HybridGap],
    max_additional: int
) -> List[float]:
    """
    fps 증가 없이 선택적으로 프레임 추가
    
    목표: fps=3 전체 추가보다 적은 프레임으로 동등/우수 성능
    
    Args:
        base_timestamps: 기본 timestamp (fps=2)
        gaps: Priority 정렬된 gap 리스트
        max_additional: 최대 추가 프레임 수
        
    Returns:
        확장된 timestamp 리스트
    """
    additional = []
    
    for gap in gaps[:max_additional]:
        new_ts = gap.midpoint_timestamp
        if new_ts not in base_timestamps:
            additional.append(new_ts)
    
    return sorted(base_timestamps + additional)
```

### 실험 설계

```
Baseline A: fps=2 (N장)           → SfM points X, PSNR P1
Baseline B: fps=3 (1.5N장)        → SfM points Y, PSNR P2

Proposed:   fps=2 + M장 추가      → SfM points Z, PSNR P3
            (M << 0.5N, 예: 0.2N)

성공 조건: Z ≥ Y and P3 ≥ P2 with (N+M) < 1.5N
```

---

## 체크리스트

- [ ] `sfm_quality_analyzer.py` 구현
- [ ] `hybrid_gap_analyzer.py` 구현  
- [ ] `frame_replacer.py` 구현
- [ ] `hybrid_pipeline.py` 구현
- [ ] 단위 테스트 작성
- [ ] Museum_cut 데이터로 Pass 1 실행
- [ ] Hybrid Pass 2 실행
- [ ] SfM points 비교 분석
- [ ] 3DGS 학습 및 PSNR 평가
- [ ] Contribution 2를 위한 선택적 추가 실험

---

## 참고 자료

- [COLMAP Output Format](https://colmap.github.io/format.html)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- 기존 구현: `CLAUDE.md`, `adaptive_sampler.py`, `trajectory_analyzer.py`

