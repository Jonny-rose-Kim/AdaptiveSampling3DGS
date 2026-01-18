# 3DGS Adaptive Frame Sampling 프로젝트

## 프로젝트 개요

본 프로젝트는 COLMAP 카메라 pose 정보를 활용하여 동영상에서 지능적으로 프레임을 추출하는 3D Gaussian Splatting(3DGS)용 적응형 프레임 샘플링 시스템을 구현합니다. 고정 fps 추출 방식 대신, translation과 rotation을 기반으로 균등한 카메라 간격을 보장하는 2-pass 파이프라인을 통해 COLMAP feature matching 성공률과 최종 3DGS 품질을 향상시킵니다.

## 문제 정의

표준 ffmpeg 고정 fps 추출 방식의 문제점:
- **불균등한 카메라 간격**: 카메라가 빠르게 움직이는 구간에서는 큰 간격이 발생하고, 느리게 움직이는 구간에서는 중복 프레임 생성
- **COLMAP 매칭 실패**: 프레임 간 카메라 거리가 먼 경우 feature point 매칭 성공률 감소
- **3DGS 품질 저하**: sparse한 커버리지 영역에서 novel view 렌더링 시 floating artifact와 noise 발생
- **비효율적인 데이터 활용**: 연속적인 비디오 정보의 비효율적 샘플링

## 솔루션 아키텍처

### 2-Pass 파이프라인

**Pass 1: 분석 단계**
```
Video → ffmpeg (최대 fps) → COLMAP → 카메라 Trajectory 분석
```

**Adaptive Sampling 계산**
```
Translation + Rotation 기반 점수화 → 최적 timestamp 계산
```

**Pass 2: 재추출 단계**
```
Video → ffmpeg (adaptive timestamps) → COLMAP → 3DGS 학습
```

## 핵심 모듈

### 모듈 구조

| 모듈 | 목적 |
|------|------|
| `FrameExtractor` | ffmpeg 래퍼, 고정 fps 및 특정 timestamp 기반 추출 지원 |
| `COLMAPParser` | COLMAP 출력 (images.txt, cameras.txt) 파싱 및 카메라 pose 추출 |
| `TrajectoryAnalyzer` | 카메라 trajectory 분석, translation/rotation 기반 거리 계산 |
| `AdaptiveSampler` | 불균등 구간 식별, 최적 샘플링 timestamp 계산 |
| `PipelineOrchestrator` | 전체 파이프라인 조율, COLMAP/3DGS 실행 관리 |

## 구현 세부사항

### Phase 1: 초기 추출 및 분석

1. **최대 fps로 프레임 추출**: 비디오 원본 fps를 사용하여 COLMAP 성공률 극대화
2. **COLMAP 실행**: feature extraction → matching → sparse reconstruction
3. **카메라 Pose 추출**: images.txt에서 quaternion(회전) + translation(위치) 파싱
4. **프레임-타임스탬프 매핑**: 파일명 기반으로 각 카메라의 원본 비디오 timestamp 복원

### Phase 2: Adaptive Sampling 알고리즘

**카메라 간 거리 메트릭**:
```
Score(i, i+1) = α × ||t[i+1] - t[i]|| + β × θ(R[i], R[i+1])
```
여기서:
- α, β: 가중치 하이퍼파라미터
- θ: rotation matrix 간 geodesic distance (각도 거리)

**균등 샘플링 전략**:
1. trajectory를 따라 누적 score 계산
2. 목표 간격 설정: `target_distance = total_score / desired_frame_count`
3. `target_distance`의 배수 지점에서 프레임 선택
4. 샘플링 지점이 프레임 사이에 있을 경우 가장 가까운 timestamp 또는 보간된 timestamp 사용

### Phase 3: 재추출 및 최종 처리

- 계산된 adaptive timestamps로 ffmpeg를 통해 프레임 재추출
- 재추출된 이미지로 COLMAP 재실행
- 3DGS 학습 수행

## 예외 처리

### COLMAP Registration 실패
- **Gap Detection**: registered 프레임 간 timestamp gap 감지
- **Dense Sampling**: 실패 구간에서 2-4배 denser하게 프레임 추출
- **Fallback**: 여전히 실패 시 선형 보간 사용 또는 해당 구간 제외

### 비디오 FPS 한계
- 원본 비디오 fps 이상으로는 프레임 추출 불가
- adaptive sampling이 더 높은 밀도를 요구하는 경우 원본 fps로 제한하고 품질 경고 로깅

### 제자리 회전 (In-place Rotation)
- 제자리에서 회전하는 카메라는 translation은 0이지만 view 변화는 큼
- score 함수의 rotation 성분(β)이 이 시나리오를 처리
- 파노라마 촬영과 같은 경우에도 적절한 샘플링 보장

## 평가 메트릭

### 정량적 평가 지표

| 지표 | 설명 |
|------|------|
| PSNR | Peak Signal-to-Noise Ratio (높을수록 좋음) |
| SSIM | Structural Similarity Index (높을수록 좋음) |
| LPIPS | Learned Perceptual Image Patch Similarity (낮을수록 좋음) |
| Camera Coverage | 카메라 간 거리 분포의 표준편차 (낮을수록 균등) |
| Registration Rate | COLMAP에서 성공적으로 등록된 카메라 비율 |

### 실험 설계

1. **Baseline (A)**: 고정 fps 추출 (ffmpeg fps=2)
2. **Proposed (B)**: Adaptive sampling 방식
3. **통제 조건**: 동일한 총 이미지 수, 동일한 3DGS 학습 설정

## 개발 가이드라인

### 코드 구조

예상 구조:
```
adaptive_sampling/
├── frame_extractor.py      # ffmpeg 래퍼
├── colmap_parser.py         # COLMAP 출력 파싱
├── trajectory_analyzer.py   # 카메라 trajectory 분석
├── adaptive_sampler.py      # Adaptive sampling 알고리즘
├── pipeline.py              # PipelineOrchestrator
├── utils/
│   ├── rotation_utils.py    # Rotation matrix 연산
│   └── metrics.py           # 평가 메트릭
└── tests/
    └── ...
```

### 코딩 표준

- **Python 3.8+**: 모든 함수 시그니처에 type hint 사용
- **NumPy/SciPy**: rotation 계산 (quaternion, geodesic distance)
- **모듈형 설계**: 각 모듈은 독립적으로 테스트 가능해야 함
- **로깅**: Python logging 모듈을 사용하여 파이프라인 진행 상황과 경고 기록
- **설정**: 하이퍼파라미터(α, β, 목표 프레임 수)를 위한 JSON/YAML 설정 파일 지원

### 기존 3DGS 파이프라인과의 통합

- 기존 `convert.py` 워크플로우와 호환되어야 함
- 출력 형식은 예상되는 COLMAP 데이터셋 구조와 일치해야 함
- 기존 학습 스크립트와의 호환성 유지

## 기술 스택

| 구분 | 도구/라이브러리 |
|------|----------------|
| 언어 | Python 3.8+ |
| 프레임 추출 | ffmpeg |
| Structure from Motion | COLMAP |
| 수치 계산 | NumPy, SciPy (rotation 계산) |
| 3D Reconstruction | 3D Gaussian Splatting |
| 평가 | PyTorch (LPIPS), scikit-image (PSNR, SSIM) |

## 기대 효과

- **품질 향상**: 균등한 카메라 간격으로 인해 sparse 구간의 floating artifact 감소
- **COLMAP 안정성**: 적절한 카메라 간격 유지로 feature matching 성공률 향상
- **데이터 효율성**: 동일한 프레임 수로 더 나은 scene coverage 달성
- **자동화**: 수동 프레임 선택 없이 자동으로 최적의 프레임 세트 추출

## 개발 단계

### 1주차: COLMAP Parser
- COLMAP 출력 파일 파싱
- 카메라 pose 추출 (quaternion + translation)
- 프레임을 timestamp에 매핑

### 2주차: Trajectory Analyzer
- 거리 메트릭 구현 (translation + rotation)
- 누적 trajectory score 계산
- 불균등 간격 구간 식별

### 3주차: Adaptive Sampler
- 균등 샘플링 알고리즘 구현
- 최적 timestamp 계산
- edge case 처리 (registration 실패, fps 한계)

### 4주차: 파이프라인 통합
- 재추출을 위한 ffmpeg 통합
- PipelineOrchestrator 구현
- End-to-end 테스트

### 5주차: 평가
- Baseline vs. proposed 비교
- 품질 메트릭 계산 (PSNR, SSIM, LPIPS)
- 카메라 coverage 균등성 분석

### 6주차: 최적화
- 하이퍼파라미터 튜닝 (α, β 가중치)
- 성능 최적화
- edge case 처리

### 7주차: 문서화
- 코드 문서화
- 사용자 가이드
- 최종 릴리즈

## Claude를 위한 중요 노트

- rotation 거리 계산 구현 시 robust한 quaternion/matrix 연산을 위해 `scipy.spatial.transform.Rotation` 사용
- ffmpeg timestamp 추출은 정밀한 프레임 선택을 위해 `-ss` 파라미터 사용
- COLMAP images.txt 형식: `IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME`
- 높은 원본 fps를 가진 긴 비디오 처리 시 메모리 효율성 고려
- 다양한 카메라 움직임 패턴으로 테스트: 선형 이동, 원형 이동, 제자리 회전, 혼합 패턴

---

# 기술적 타당성 검증

## ✅ 실현 가능한 부분

### 1. COLMAP Pose 파싱
- **타당성**: COLMAP은 images.txt에 카메라 pose를 quaternion + translation으로 저장
- **구현 가능성**: 텍스트 파일 파싱은 straightforward
- **검증**: ✅ 완전히 가능

### 2. Translation Distance 계산
- **타당성**: Euclidean distance `||t[i+1] - t[i]||`는 standard 3D geometry
- **구현 가능성**: NumPy로 간단히 구현 가능
- **검증**: ✅ 완전히 가능

### 3. Rotation Distance (Geodesic Distance)
- **타당성**: Rotation matrix 간 geodesic distance는 SO(3) 매니폴드에서 표준 메트릭
- **구현 가능성**: `scipy.spatial.transform.Rotation`이 이를 직접 지원
- **검증**: ✅ 완전히 가능

### 4. ffmpeg Timestamp 기반 추출
- **타당성**: ffmpeg는 `-ss` 파라미터로 정밀한 timestamp 추출 지원
- **구현 가능성**: 계산된 timestamp 리스트를 순회하며 프레임 추출
- **검증**: ✅ 완전히 가능

### 5. 2-Pass Pipeline 구조
- **타당성**: 1차 pass로 카메라 trajectory 분석 후 2차 pass로 재추출하는 논리적 흐름
- **구현 가능성**: 각 단계가 독립적으로 실행 가능
- **검증**: ✅ 완전히 가능

## ⚠️ 잠재적 문제점 및 해결 방안

### 1. 프레임-Timestamp 매핑의 정확성
**문제**:
- 1차 추출된 프레임의 파일명에서 timestamp를 복원해야 하는데, ffmpeg 추출 시 정확한 timestamp 정보가 보존되지 않을 수 있음
- 프레임 번호와 timestamp의 정확한 매핑이 필요

**해결 방안**:
- ffmpeg 추출 시 `-frame_pts true` 옵션 사용하여 PTS(Presentation Timestamp) 정보 보존
- 또는 추출 시 timestamp를 파일명에 명시적으로 포함: `frame_%06d_%010.3f.png` (프레임번호_timestamp)
- metadata 파일을 별도로 생성하여 프레임-timestamp 매핑 저장

**수정 제안**: Phase 1에 명시적 timestamp 기록 단계 추가

### 2. COLMAP Registration 순서 불일치
**문제**:
- COLMAP은 입력 프레임 순서대로 카메라를 등록하지 않을 수 있음
- Sequential extraction 가정이 깨질 수 있음

**해결 방안**:
- images.txt에서 IMAGE_ID와 파일명을 함께 파싱하여 매핑 테이블 생성
- 파일명의 timestamp 정보를 기반으로 재정렬
- Registration되지 않은 프레임은 제외하고 trajectory 계산

**검증**: ⚠️ 구현 복잡도 증가하지만 해결 가능

### 3. Video Seeking 정확도
**문제**:
- ffmpeg의 `-ss` 파라미터는 keyframe seeking으로 인해 정확하지 않을 수 있음
- 요청한 timestamp와 실제 추출된 프레임 간 오차 발생 가능

**해결 방안**:
- `-ss` 옵션을 input 옵션으로 사용 시 빠르지만 부정확 (keyframe seeking)
- `-ss` 옵션을 output 옵션으로 사용하거나 `-accurate_seek` 플래그 추가하여 정확도 향상
- 명령어 예: `ffmpeg -accurate_seek -ss {timestamp} -i video.mp4 -frames:v 1 output.png`

**검증**: ⚠️ 성능과 정확도 trade-off 필요하지만 해결 가능

### 4. Score 함수의 하이퍼파라미터 (α, β) 선택
**문제**:
- Translation과 rotation의 단위가 다름 (meter vs. radian)
- 적절한 α, β 값 선택이 어려울 수 있음

**해결 방안**:
- Translation과 rotation을 각각 정규화 (normalize)
- 정규화 방법: 전체 trajectory에서의 최대값으로 나누기
- 또는 평균과 표준편차로 z-score normalization
- α=β=0.5로 시작하여 실험적으로 조정

**수정 제안**:
```python
# Normalized score
trans_norm = ||t[i+1] - t[i]|| / max_translation
rot_norm = θ(R[i], R[i+1]) / max_rotation
Score(i, i+1) = α × trans_norm + β × rot_norm
```

**검증**: ⚠️ 실험적 튜닝 필요하지만 논리적으로 타당

### 5. Dense Sampling in Failed Regions
**문제**:
- COLMAP 실패 구간을 어떻게 식별하고 dense sampling을 적용할지 불명확
- 2-4배 denser 샘플링의 기준이 모호

**해결 방안**:
- 1차 COLMAP에서 registration된 프레임의 timestamp를 분석
- Consecutive registered frames 간 timestamp gap이 평균의 2배 이상인 경우 "실패 구간"으로 식별
- 해당 구간에서 원본 비디오의 모든 프레임 추출 또는 더 작은 timestamp 간격으로 추출

**검증**: ⚠️ 휴리스틱이지만 논리적으로 타당

### 6. Interpolated Timestamp 처리
**문제**:
- "가장 가까운 timestamp 또는 보간된 timestamp 사용"이 모호함
- ffmpeg는 존재하지 않는 timestamp로 프레임을 생성할 수 없음

**해결 방안**:
- **실제로는 보간이 아닌 nearest neighbor 방식 사용**
- 계산된 이상적인 timestamp에 가장 가까운 실제 비디오 프레임을 선택
- Video fps를 고려하여 가능한 timestamp 리스트 생성 후 nearest matching

**수정 제안**: "보간된 timestamp" 표현을 "가장 가까운 실제 프레임 timestamp"로 수정

## 🔧 권장 수정사항

### Phase 1 수정
```
1. 최대 fps로 프레임 추출 + timestamp 메타데이터 기록
   - ffmpeg 추출 시 timestamp를 파일명 또는 별도 메타데이터 파일에 기록

2. COLMAP 실행

3. 카메라 Pose + 프레임-Timestamp 매핑 추출
   - images.txt에서 등록된 프레임의 IMAGE_ID, 파일명, pose 추출
   - 파일명/메타데이터에서 timestamp 복원
   - (IMAGE_ID, timestamp, pose) 튜플 생성
```

### Phase 2 수정
```
카메라 간 거리 메트릭 (정규화 버전):
trans_dist = ||t[i+1] - t[i]||
rot_dist = θ(R[i], R[i+1])

trans_normalized = trans_dist / max(all_trans_dists)
rot_normalized = rot_dist / max(all_rot_dists)

Score(i, i+1) = α × trans_normalized + β × rot_normalized
```

### Phase 3 수정
```
균등 샘플링 전략:
1. 누적 score 계산
2. target_distance 설정
3. 샘플링 포인트 결정
4. 각 샘플링 포인트에 대해:
   - 계산된 이상적 timestamp에서 가장 가까운 실제 비디오 프레임 timestamp 찾기
   - Video fps를 고려하여 가능한 timestamp: [0, 1/fps, 2/fps, ..., duration]
```

## 📊 최종 검증 결과

| 항목 | 타당성 | 위험도 | 비고 |
|------|--------|--------|------|
| COLMAP 파싱 | ✅ | 낮음 | 표준 텍스트 파싱 |
| Translation 거리 | ✅ | 낮음 | 기본 3D geometry |
| Rotation 거리 | ✅ | 낮음 | SciPy 지원 |
| Score 함수 | ⚠️ | 중간 | 정규화 및 α,β 튜닝 필요 |
| ffmpeg 추출 | ⚠️ | 중간 | Seeking 정확도 이슈, 해결 가능 |
| Timestamp 매핑 | ⚠️ | 중간 | 명시적 기록 메커니즘 필요 |
| 2-Pass Pipeline | ✅ | 낮음 | 논리적으로 타당 |
| Dense Sampling | ⚠️ | 중간 | 휴리스틱이지만 합리적 |
| 전체 시스템 | ✅ | 중간 | 구현 가능, 세부 조정 필요 |

## 결론

**✅ 프로젝트는 기술적으로 실현 가능합니다.**

주요 권장사항:
1. **Timestamp 기록 메커니즘 명확화**: 1차 추출 시 명시적 timestamp 기록
2. **Score 함수 정규화**: Translation과 rotation을 정규화하여 α, β 튜닝 용이하게
3. **ffmpeg Seeking 정확도**: `-accurate_seek` 옵션 사용
4. **"보간" 표현 수정**: "가장 가까운 실제 프레임"으로 명확히
5. **실험적 검증 필요**: 하이퍼파라미터 튜닝을 위한 소규모 테스트 데이터셋 준비

프로젝트의 핵심 아이디어는 탄탄하며, 위 수정사항들을 반영하면 성공적으로 구현할 수 있습니다.
