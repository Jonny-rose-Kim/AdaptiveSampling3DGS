# 3DGS Adaptive Frame Sampling

COLMAP 카메라 pose 기반 동적 프레임 추출 시스템

## 개요

3D Gaussian Splatting을 위한 adaptive frame sampling 시스템입니다. 고정 fps로 프레임을 추출하는 기존 방식 대신, COLMAP 카메라 pose 정보를 분석하여 카메라 움직임에 따라 adaptive하게 프레임을 재추출합니다.

### 주요 기능

- **2-Pass Pipeline**: 초기 추출 → 분석 → 재추출
- **Trajectory 분석**: Translation + Rotation 기반 거리 메트릭
- **Adaptive Sampling**: 카메라 간 균등한 간격 유지
- **Sparse 구간 처리**: 큰 gap이 있는 구간 자동 감지 및 densification

## 설치

### 필수 요구사항

- Python 3.8+
- ffmpeg (비디오 프레임 추출)
- COLMAP (Structure from Motion)

### Python 패키지 설치

```bash
pip install numpy scipy
```

## 사용법

### 1. 기본 사용 (Python API)

```python
from adaptive_sampling.pipeline import PipelineOrchestrator

# Pipeline 생성
pipeline = PipelineOrchestrator(
    video_path="video.mp4",
    workspace_dir="./workspace",
    alpha=0.5,  # translation 가중치
    beta=0.5,   # rotation 가중치
    normalize=True
)

# 전체 파이프라인 실행
result = pipeline.run_full_pipeline(
    desired_frame_count=100,  # 최종 프레임 수
    pass1_fps=30  # Pass 1 추출 fps
)

print(f"최종 프레임: {result['pass2_frames']}개")
print(f"출력 디렉토리: {result['pass2_dir']}")
```

### 2. 단계별 실행

#### Pass 1: 초기 프레임 추출 및 COLMAP

```python
from adaptive_sampling.frame_extractor import FrameExtractor
from adaptive_sampling.colmap_parser import COLMAPParser

# 1. 프레임 추출
extractor = FrameExtractor("video.mp4")
files, metadata = extractor.extract_frames_with_metadata(
    output_dir="./pass1/images",
    fps=30
)

# 2. COLMAP 실행 (외부 스크립트)
# colmap feature_extractor --database_path database.db --image_path pass1/images/
# colmap exhaustive_matcher --database_path database.db
# colmap mapper --database_path database.db --image_path pass1/images/ --output_path pass1/sparse/

# 3. COLMAP 결과 파싱
parser = COLMAPParser("./pass1/sparse/0")
poses = parser.parse_and_extract(fps=30)
```

#### Trajectory 분석 및 Adaptive Sampling

```python
from adaptive_sampling.trajectory_analyzer import TrajectoryAnalyzer
from adaptive_sampling.adaptive_sampler import AdaptiveSampler

# 1. Trajectory 분석
analyzer = TrajectoryAnalyzer(alpha=0.5, beta=0.5, normalize=True)
segments = analyzer.analyze_trajectory(poses)

# 통계
stats = analyzer.get_statistics(segments)
print(f"Total score: {stats['total_score']:.3f}")
print(f"Translation mean: {stats['translation']['mean']:.3f}")

# 2. Sparse 구간 식별
sparse_regions = analyzer.identify_sparse_regions(segments, threshold_multiplier=2.0)
print(f"Sparse regions: {sparse_regions}")

# 3. Adaptive sampling
sampler = AdaptiveSampler(video_fps=30)
timestamps = sampler.compute_target_timestamps(segments, desired_frame_count=100)
print(f"Computed {len(timestamps)} timestamps")
```

#### Pass 2: Adaptive 재추출

```python
# Adaptive timestamp로 프레임 재추출
extracted_files = extractor.extract_frames_by_timestamps(
    timestamps,
    output_dir="./pass2/images",
    accurate_seek=True
)

print(f"Extracted {len(extracted_files)} frames")
```

### 3. 설정 파일 사용

```python
from adaptive_sampling.config import AdaptiveSamplingConfig

# 설정 로드 또는 생성
config = AdaptiveSamplingConfig(
    alpha=0.6,
    beta=0.4,
    normalize=True,
    sparse_threshold_multiplier=2.5
)

# 설정 저장
config.save("my_config.json")

# 설정 로드
config = AdaptiveSamplingConfig.load("my_config.json")
```

## 모듈 구조

```
adaptive_sampling/
├── __init__.py
├── colmap_parser.py         # COLMAP 출력 파싱
├── trajectory_analyzer.py   # Trajectory 분석
├── adaptive_sampler.py      # Adaptive sampling 알고리즘
├── frame_extractor.py       # ffmpeg 래퍼
├── pipeline.py              # 전체 파이프라인 조율
├── config.py                # 설정 관리
├── utils/
│   ├── __init__.py
│   └── rotation_utils.py    # Rotation 계산 유틸리티
└── tests/
    ├── test_colmap_parser.py
    ├── test_trajectory_analyzer.py
    └── test_adaptive_sampler.py
```

## 알고리즘 설명

### Score 함수

카메라 간 거리를 다음과 같이 정의합니다:

```
Score(i, i+1) = α × ||t[i+1] - t[i]|| + β × θ(R[i], R[i+1])
```

여기서:
- `t`: translation vector
- `θ(R[i], R[i+1])`: rotation matrix 간 geodesic distance (radians)
- `α, β`: 가중치 하이퍼파라미터

정규화 사용 시:

```
trans_norm = ||t[i+1] - t[i]|| / max_translation
rot_norm = θ(R[i], R[i+1]) / max_rotation
Score(i, i+1) = α × trans_norm + β × rot_norm
```

### Adaptive Sampling 전략

1. **누적 score 계산**: trajectory를 따라 누적 score 계산
2. **목표 간격 설정**: `target_distance = total_score / (desired_frame_count - 1)`
3. **샘플링 포인트 결정**: 누적 거리가 `target_distance`의 배수가 되는 지점에서 프레임 선택
4. **Timestamp 스냅**: 가장 가까운 실제 비디오 프레임 timestamp 사용

### Sparse 구간 처리

1. **감지**: score가 평균의 N배(기본 2배) 이상인 구간 식별
2. **Densification**: 해당 구간에서 추가 프레임 추출 (2-4배)

## 테스트

```bash
# 전체 테스트 실행
python -m pytest adaptive_sampling/tests/ -v

# 개별 모듈 테스트
python adaptive_sampling/tests/test_colmap_parser.py
python adaptive_sampling/tests/test_trajectory_analyzer.py
python adaptive_sampling/tests/test_adaptive_sampler.py
```

## 예상 성능 향상

- **품질 향상**: Sparse 구간의 floating artifact 감소
- **COLMAP 안정성**: 적절한 카메라 간격으로 feature matching 성공률 향상
- **데이터 효율성**: 동일한 프레임 수로 더 나은 scene coverage

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 라이선스

MIT License

## 참고

- Original 3DGS implementation: https://github.com/graphdeco-inria/gaussian-splatting
- COLMAP: https://colmap.github.io/
- Plan document: `plan.md`
