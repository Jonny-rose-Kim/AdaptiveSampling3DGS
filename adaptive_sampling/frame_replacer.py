"""
Frame Replacer

기여도 낮은 프레임을 높은 priority gap의 새 timestamp로 교체
"""

from typing import List, Set, Optional
from dataclasses import dataclass
import numpy as np

try:
    from .sfm_quality_analyzer import ImageContribution
    from .hybrid_gap_analyzer import HybridGap
except ImportError:
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
        existing_ts_copy = existing_timestamps.copy()

        for img in low_contrib_images:
            # 아직 사용하지 않은 gap 중 가장 높은 priority 선택
            best_gap = None
            for gap in high_priority_gaps:
                gap_key = (gap.start_timestamp, gap.end_timestamp)
                if gap_key not in used_gaps:
                    best_gap = gap
                    break

            if best_gap is None:
                print(f"Warning: No available gap for {img.image_name}")
                continue

            # Gap 중간 지점으로 새 timestamp 결정
            new_ts = self._snap_to_video_frame(best_gap.midpoint_timestamp)

            # 이미 존재하는 timestamp면 약간 조정
            attempts = 0
            max_attempts = 10
            original_new_ts = new_ts
            while new_ts in existing_ts_copy and attempts < max_attempts:
                # 원본 중간점에서 ±1, ±2, ... 프레임 이동
                offset = (attempts // 2 + 1) * (1 if attempts % 2 == 0 else -1)
                new_ts = self._snap_to_video_frame(
                    original_new_ts + offset / self.video_fps
                )
                attempts += 1

            if new_ts in existing_ts_copy:
                print(f"Warning: Could not find unique timestamp for gap "
                      f"[{best_gap.start_timestamp:.2f}, {best_gap.end_timestamp:.2f}]")
                continue

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
            existing_ts_copy.add(new_ts)
            existing_ts_copy.discard(img.timestamp)

        return replacements

    def _snap_to_video_frame(self, timestamp: float) -> float:
        """비디오 fps에 맞는 가장 가까운 timestamp로 스냅"""
        frame_interval = 1.0 / self.video_fps
        frame_number = round(timestamp / frame_interval)
        return round(frame_number * frame_interval, 6)  # 부동소수점 오차 방지

    def _generate_reason(
        self,
        img: ImageContribution,
        gap: HybridGap
    ) -> str:
        """교체 근거 문자열 생성"""
        return (
            f"Low contribution ({img.contribution_score:.0f} points) -> "
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

    def print_replacement_report(
        self,
        replacements: List[FrameReplacement],
        show_all: bool = False
    ) -> None:
        """교체 보고서 출력"""
        print("\n" + "=" * 70)
        print("FRAME REPLACEMENT REPORT")
        print("=" * 70)

        if not replacements:
            print("No replacements planned.")
            return

        # 요약 통계
        removed_contributions = [r.remove_contribution for r in replacements]
        gap_priorities = [r.target_gap.gap_priority for r in replacements]

        print(f"\nSummary:")
        print(f"  Total replacements: {len(replacements)}")
        print(f"  Removed contribution: mean={np.mean(removed_contributions):.1f}, "
              f"range=[{np.min(removed_contributions):.0f}, {np.max(removed_contributions):.0f}]")
        print(f"  Gap priority: mean={np.mean(gap_priorities):.3f}, "
              f"range=[{np.min(gap_priorities):.3f}, {np.max(gap_priorities):.3f}]")

        # 상세 목록 (show_all이면 전체, 아니면 상위 5개)
        display_count = len(replacements) if show_all else min(5, len(replacements))
        print(f"\nReplacement Details (showing {display_count}/{len(replacements)}):")
        print("-" * 70)

        for i, r in enumerate(replacements[:display_count]):
            print(f"\n[{i+1}] {r.remove_image_name}")
            print(f"    Remove: t={r.remove_timestamp:.3f}s "
                  f"(contribution={r.remove_contribution:.0f})")
            print(f"    Add:    t={r.new_timestamp:.3f}s")
            print(f"    Gap:    [{r.target_gap.start_timestamp:.3f}s ~ "
                  f"{r.target_gap.end_timestamp:.3f}s]")
            print(f"    Reason: {r.reason}")

        if not show_all and len(replacements) > display_count:
            print(f"\n... and {len(replacements) - display_count} more replacements")

        print("\n" + "=" * 70)

    def get_replacement_summary(self, replacements: List[FrameReplacement]) -> dict:
        """교체 정보 요약 딕셔너리 반환"""
        if not replacements:
            return {
                'count': 0,
                'removed_contribution': {},
                'gap_priority': {},
                'replacements': []
            }

        removed_contributions = [r.remove_contribution for r in replacements]
        gap_priorities = [r.target_gap.gap_priority for r in replacements]

        return {
            'count': len(replacements),
            'removed_contribution': {
                'mean': float(np.mean(removed_contributions)),
                'min': float(np.min(removed_contributions)),
                'max': float(np.max(removed_contributions)),
            },
            'gap_priority': {
                'mean': float(np.mean(gap_priorities)),
                'min': float(np.min(gap_priorities)),
                'max': float(np.max(gap_priorities)),
            },
            'replacements': [
                {
                    'remove_timestamp': r.remove_timestamp,
                    'remove_image_name': r.remove_image_name,
                    'remove_contribution': r.remove_contribution,
                    'new_timestamp': r.new_timestamp,
                    'gap_start': r.target_gap.start_timestamp,
                    'gap_end': r.target_gap.end_timestamp,
                    'gap_priority': r.target_gap.gap_priority,
                    'reason': r.reason,
                }
                for r in replacements
            ]
        }


if __name__ == "__main__":
    print("FrameReplacer module - use via hybrid_pipeline.py")
