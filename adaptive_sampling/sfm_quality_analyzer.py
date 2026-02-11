"""
SfM Quality Analyzer

COLMAP 출력을 분석하여 각 이미지의 SfM 기여도를 계산합니다.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import re


@dataclass
class ImageContribution:
    """이미지의 SfM 기여도 정보"""
    image_id: int
    image_name: str
    timestamp: float

    # 기여도 지표
    total_observations: int      # 전체 2D point 수
    valid_observations: int      # 유효한 3D point 관측 수 (POINT3D_ID != -1)

    # 추가 분석용
    observed_point3d_ids: Set[int] = field(default_factory=set)

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

    def __init__(self, colmap_dir: str, extraction_fps: float = 2.0):
        """
        Args:
            colmap_dir: COLMAP sparse reconstruction 디렉토리 경로
                       (보통 <dataset>/sparse/0/)
            extraction_fps: 프레임 추출에 사용된 fps (timestamp 계산용)
        """
        self.colmap_dir = Path(colmap_dir)
        self.extraction_fps = extraction_fps

        # 파일 경로 설정
        self.images_txt = self.colmap_dir / "images.txt"
        self.images_bin = self.colmap_dir / "images.bin"
        self.points3d_txt = self.colmap_dir / "points3D.txt"
        self.points3d_bin = self.colmap_dir / "points3D.bin"

    def _ensure_txt_files(self) -> bool:
        """
        .txt 파일이 없으면 .bin에서 변환 시도

        Returns:
            True if txt files are available
        """
        if self.images_txt.exists():
            return True

        if self.images_bin.exists():
            import subprocess
            print(f"Converting binary to text format...")
            result = subprocess.run([
                'colmap', 'model_converter',
                '--input_path', str(self.colmap_dir),
                '--output_path', str(self.colmap_dir),
                '--output_type', 'TXT'
            ], capture_output=True, text=True)

            if result.returncode == 0 and self.images_txt.exists():
                print("Conversion successful")
                return True
            else:
                print(f"Conversion failed: {result.stderr}")
                return False

        return False

    def parse_images_with_observations(self) -> Dict[int, ImageContribution]:
        """
        images.txt를 파싱하여 각 이미지의 기여도 정보를 추출

        Returns:
            image_id를 key로 하는 ImageContribution 딕셔너리
        """
        if not self._ensure_txt_files():
            raise FileNotFoundError(f"images.txt not found and conversion failed: {self.images_txt}")

        contributions = {}

        with open(self.images_txt, 'r') as f:
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

                # 파일명에서 timestamp 추출
                timestamp = self._extract_timestamp(image_name)

                # 둘째 줄: POINTS2D[] as (X, Y, POINT3D_ID) ...
                i += 1
                if i >= len(lines):
                    break

                points_line = lines[i].strip()
                point3d_ids = self._parse_point3d_ids(points_line)

                # 유효한 관측 계산 (POINT3D_ID != -1)
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

    def _extract_timestamp(self, filename: str) -> float:
        """파일명에서 timestamp 추출"""
        # frame_NNNNNN.ext 패턴
        match = re.search(r'frame_(\d+)', filename)
        if match:
            frame_number = int(match.group(1))
            return frame_number / self.extraction_fps

        # NNNNNN.ext 패턴
        match = re.search(r'(\d+)\.', filename)
        if match:
            frame_number = int(match.group(1))
            return frame_number / self.extraction_fps

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
        bottom_ratio: float = 0.2,
        protect_edge_frames: int = 5
    ) -> List[ImageContribution]:
        """
        기여도 하위 K장 선택 (시작/끝 프레임 보호)

        Args:
            contributions: 이미지 기여도 딕셔너리
            bottom_ratio: 하위 비율 (0.2 = 하위 20%)
            protect_edge_frames: 보호할 시작/끝 프레임 수

        Returns:
            기여도 낮은 이미지 리스트 (기여도 오름차순)
        """
        # timestamp 순으로 정렬
        sorted_by_time = sorted(
            contributions.values(),
            key=lambda x: x.timestamp
        )

        # 시작/끝 프레임 ID 수집 (보호 대상)
        edge_frame_ids = set()
        if protect_edge_frames > 0:
            edge_frame_ids.update(
                img.image_id for img in sorted_by_time[:protect_edge_frames]
            )
            edge_frame_ids.update(
                img.image_id for img in sorted_by_time[-protect_edge_frames:]
            )

        # edge 프레임 제외하고 기여도 순으로 정렬
        inner_images = [
            img for img in contributions.values()
            if img.image_id not in edge_frame_ids
        ]
        sorted_by_contribution = sorted(
            inner_images,
            key=lambda x: x.contribution_score
        )

        # 하위 K개 선택
        total_count = len(contributions)
        k = int(total_count * bottom_ratio)
        k = max(1, min(k, len(sorted_by_contribution)))  # 최소 1, 최대 inner_images 수

        return sorted_by_contribution[:k]

    def get_statistics(
        self,
        contributions: Dict[int, ImageContribution]
    ) -> dict:
        """기여도 통계 정보"""
        if not contributions:
            return {'num_images': 0}

        scores = [c.contribution_score for c in contributions.values()]
        ratios = [c.observation_ratio for c in contributions.values()]

        return {
            'num_images': len(contributions),
            'contribution': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
            },
            'observation_ratio': {
                'mean': float(np.mean(ratios)),
                'std': float(np.std(ratios)),
                'min': float(np.min(ratios)),
                'max': float(np.max(ratios)),
            }
        }

    def get_continuity_statistics(
        self,
        continuities: List[FeatureTrackContinuity]
    ) -> dict:
        """연속성 통계 정보"""
        if not continuities:
            return {'num_pairs': 0}

        shared_counts = [c.shared_point_count for c in continuities]
        scores = [c.continuity_score for c in continuities]

        return {
            'num_pairs': len(continuities),
            'shared_points': {
                'mean': float(np.mean(shared_counts)),
                'std': float(np.std(shared_counts)),
                'min': int(np.min(shared_counts)),
                'max': int(np.max(shared_counts)),
            },
            'continuity_score': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
            }
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sfm_quality_analyzer.py <colmap_dir> [extraction_fps]")
        print("Example: python sfm_quality_analyzer.py ./data/Museum_cut_exp/pass1/sparse/0 2.0")
        sys.exit(1)

    colmap_dir = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    print("=" * 60)
    print("SfM Quality Analyzer")
    print("=" * 60)
    print(f"COLMAP dir: {colmap_dir}")
    print(f"Extraction FPS: {fps}")

    analyzer = SfMQualityAnalyzer(colmap_dir, extraction_fps=fps)

    # 기여도 분석
    print("\n[1] Parsing images with observations...")
    contributions = analyzer.parse_images_with_observations()
    stats = analyzer.get_statistics(contributions)

    print(f"  Total images: {stats['num_images']}")
    print(f"  Contribution: mean={stats['contribution']['mean']:.1f}, "
          f"std={stats['contribution']['std']:.1f}")
    print(f"  Range: [{stats['contribution']['min']:.0f}, {stats['contribution']['max']:.0f}]")

    # 연속성 분석
    print("\n[2] Computing feature track continuity...")
    continuities = analyzer.compute_feature_track_continuity(contributions)
    cont_stats = analyzer.get_continuity_statistics(continuities)

    print(f"  Total pairs: {cont_stats['num_pairs']}")
    print(f"  Shared points: mean={cont_stats['shared_points']['mean']:.1f}, "
          f"range=[{cont_stats['shared_points']['min']}, {cont_stats['shared_points']['max']}]")
    print(f"  Continuity score: mean={cont_stats['continuity_score']['mean']:.3f}")

    # 하위 기여도 이미지 선택
    print("\n[3] Selecting low contribution images (bottom 20%)...")
    low_contrib = analyzer.select_low_contribution_images(contributions, bottom_ratio=0.2)

    print(f"  Selected: {len(low_contrib)} images")
    if low_contrib:
        print(f"  Contribution range: [{low_contrib[0].contribution_score:.0f}, "
              f"{low_contrib[-1].contribution_score:.0f}]")
        print(f"\n  Top 5 lowest:")
        for i, img in enumerate(low_contrib[:5]):
            print(f"    {i+1}. {img.image_name}: {img.contribution_score:.0f} points "
                  f"(ratio={img.observation_ratio:.2%})")

    print("\n" + "=" * 60)
    print("Done!")
