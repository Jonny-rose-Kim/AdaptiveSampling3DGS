"""
Configuration management for Adaptive Sampling
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Optional


@dataclass
class AdaptiveSamplingConfig:
    """Adaptive Sampling 설정"""

    # Trajectory Analysis
    alpha: float = 0.5  # Translation 가중치
    beta: float = 0.5  # Rotation 가중치
    normalize: bool = True  # Score 정규화 여부

    # Sampling
    sparse_threshold_multiplier: float = 2.0  # Sparse 구간 감지 threshold
    densification_factor: int = 2  # Sparse 구간 densification 배수

    # Frame Extraction
    pass1_fps: Optional[float] = None  # Pass 1 fps (None = 원본 fps)
    image_quality: int = 2  # JPEG 품질 (2=최고, 31=최저)
    accurate_seek: bool = True  # 정확한 seeking 사용

    # COLMAP
    colmap_matcher: str = "exhaustive"  # exhaustive, sequential, vocab_tree
    colmap_camera_model: str = "PINHOLE"  # PINHOLE, RADIAL, OPENCV

    def save(self, path: str):
        """설정을 JSON 파일로 저장"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'AdaptiveSamplingConfig':
        """JSON 파일에서 설정 로드"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def get_default(cls) -> 'AdaptiveSamplingConfig':
        """기본 설정 반환"""
        return cls()

    def __repr__(self):
        """설정 출력"""
        lines = ["AdaptiveSamplingConfig:"]
        lines.append(f"  Trajectory Analysis:")
        lines.append(f"    alpha (translation weight): {self.alpha}")
        lines.append(f"    beta (rotation weight): {self.beta}")
        lines.append(f"    normalize: {self.normalize}")
        lines.append(f"  Sampling:")
        lines.append(f"    sparse_threshold_multiplier: {self.sparse_threshold_multiplier}")
        lines.append(f"    densification_factor: {self.densification_factor}")
        lines.append(f"  Frame Extraction:")
        lines.append(f"    pass1_fps: {self.pass1_fps if self.pass1_fps else 'original'}")
        lines.append(f"    image_quality: {self.image_quality}")
        lines.append(f"    accurate_seek: {self.accurate_seek}")
        return '\n'.join(lines)


if __name__ == "__main__":
    # 기본 설정 생성 및 저장
    config = AdaptiveSamplingConfig.get_default()
    print(config)

    # 저장
    config.save("config_default.json")
    print(f"\nSaved to config_default.json")

    # 로드
    loaded_config = AdaptiveSamplingConfig.load("config_default.json")
    print(f"\nLoaded config:")
    print(loaded_config)
