# 🎬 Adaptive Sampling 실행 가이드

비디오 → Pass 1 (COLMAP) → Adaptive Sampling → Pass 2 전체 프로세스

---

## 📋 준비물

- ✅ 입력 비디오 파일 (예: `Museum.mp4`)
- ✅ COLMAP 설치됨
- ✅ Python 환경 (numpy, scipy 설치)
- ✅ ffmpeg 설치됨

---

## 🚀 전체 프로세스 (3단계)

### **Step 1: Pass 1 - 초기 프레임 추출**

비디오에서 고정 fps로 프레임을 추출합니다.

```bash
cd /home/jonny/jonny/Adaptive-ffmpeg

# 실행
python run_pass1.py <비디오_경로> <작업_디렉토리> [fps]

# 예시
python run_pass1.py Museum.mp4 ./museum_output 10
```

**결과**:
- `<작업_디렉토리>/pass1/input/` - 추출된 프레임들
- 비디오 정보 및 설정 출력

**소요 시간**: 5~10분 (비디오 길이에 따라)

---

### **Step 2: COLMAP - 카메라 Pose 추출**

추출된 프레임으로 COLMAP을 실행합니다.

```bash
# COLMAP 자동 실행
python run_colmap.py <작업_디렉토리>

# 예시
python run_colmap.py ./museum_output
```

**또는 기존 convert.py 사용**:
```bash
python convert.py -s <작업_디렉토리>/pass1
```

**결과**:
- `<작업_디렉토리>/pass1/sparse/0/images.txt` - 카메라 pose
- `<작업_디렉토리>/pass1/sparse/0/cameras.txt` - 카메라 파라미터

**소요 시간**: 30분~2시간 (프레임 수에 따라)

**진행 상황 확인**:
```bash
# GPU 사용 확인
nvidia-smi

# COLMAP 프로세스 확인
ps aux | grep colmap

# 로그 확인 (실행 중인 경우)
tail -f colmap_log.txt
```

---

### **Step 3: Pass 2 - Adaptive Sampling**

COLMAP 결과를 분석하고 adaptive하게 프레임을 재추출합니다.

```bash
# Pass 2 실행
python run_pass2.py <작업_디렉토리> <목표_프레임_수>

# 예시 (Pass 1의 50%로 줄이기)
python run_pass2.py ./museum_output 2852
```

**결과**:
- `<작업_디렉토리>/pass2/images/` - Adaptive하게 재추출된 프레임들
- `<작업_디렉토리>/adaptive_timestamps.json` - 계산된 timestamp
- `<작업_디렉토리>/pipeline_result.json` - 결과 요약

**소요 시간**: 10~20분

---

## 📝 간단 사용법 (Museum.mp4 예시)

### 현재 상황 (Museum.mp4)

```bash
# 1. Pass 1 완료 ✅
# - 5,704개 프레임 추출 완료

# 2. COLMAP 실행 중 🔄
# - nvidia-smi로 확인 가능 (colmap 프로세스)
# - 완료까지 대기 필요

# 3. COLMAP 완료 후
python test_museum_pass2.py
```

### 새로운 비디오로 시작하기

```bash
# 1단계: Pass 1 (프레임 추출)
python run_pass1.py my_video.mp4 ./my_output 10

# 2단계: COLMAP (대기...)
python convert.py -s ./my_output/pass1
# 또는
python run_colmap.py ./my_output

# 3단계: Pass 2 (Adaptive Sampling)
python run_pass2.py ./my_output 1000
```

---

## 🎯 Quick Start 스크립트

전체 프로세스를 한 번에 실행 (COLMAP 제외):

```bash
# 전체 파이프라인 실행
python run_full_pipeline.py <비디오> <출력_디렉토리> <최종_프레임수> [pass1_fps]

# 예시
python run_full_pipeline.py video.mp4 ./output 1000 10
```

이 스크립트는:
1. Pass 1 실행
2. COLMAP 명령어 출력 (수동 실행 필요)
3. COLMAP 완료 확인
4. Pass 2 자동 실행

---

## 📊 결과 확인

### Pass 1 vs Pass 2 비교

```bash
# 프레임 수 확인
ls pass1/input/*.png | wc -l
ls pass2/images/*.png | wc -l

# 결과 JSON 확인
cat pipeline_result.json | python -m json.tool
```

### Adaptive Sampling 효과

- **균등한 커버리지**: 카메라 간 거리 표준편차 감소
- **Sparse 구간 처리**: 큰 gap에서 더 많은 프레임 추출
- **프레임 감소**: 동일 품질로 50% 감소 가능

---

## ⚙️ 고급 설정

### 파라미터 조정

```python
# config.json 생성
{
  "alpha": 0.5,           // Translation 가중치
  "beta": 0.5,            // Rotation 가중치
  "normalize": true,      // Score 정규화
  "sparse_threshold": 2.0 // Sparse 구간 감지 threshold
}

# 설정 파일로 실행
python run_pass2.py ./output 1000 --config config.json
```

### COLMAP 옵션

```bash
# GPU 사용 (기본값)
python convert.py -s ./pass1

# CPU만 사용
python convert.py -s ./pass1 --no_gpu

# 이미지 리사이즈 (메모리 절약)
python convert.py -s ./pass1 --resize
```

---

## 🐛 문제 해결

### COLMAP 실행 확인

```bash
# COLMAP이 실행 중인지 확인
nvidia-smi  # GPU 사용 확인
ps aux | grep colmap

# COLMAP 완료 확인
ls pass1/sparse/0/images.txt
```

### 메모리 부족

- Pass 1 fps를 낮추기 (10 → 5)
- COLMAP 실행 시 `--resize` 옵션 사용
- 작은 테스트 비디오로 먼저 시도

### 에러 발생 시

```bash
# 로그 확인
cat colmap_log.txt
cat pipeline_result.json

# 작업 디렉토리 정리 후 재시작
rm -rf ./output
```

---

## 📁 출력 구조

```
<작업_디렉토리>/
├── pass1/
│   ├── input/              # Pass 1 프레임
│   ├── distorted/          # COLMAP 중간 결과
│   └── sparse/0/           # COLMAP 출력
│       ├── images.txt      # 카메라 pose ✨
│       ├── cameras.txt     # 카메라 파라미터
│       └── points3D.txt    # 3D 포인트
├── pass2/
│   └── images/             # Pass 2 프레임 ✨
├── adaptive_timestamps.json
├── pipeline_result.json
└── test_config.json
```

---

## 🎓 다음 단계

Adaptive Sampling 완료 후:

1. **3DGS 학습**:
```bash
# Pass 1 (Baseline)
python train.py -s <작업_디렉토리>/pass1

# Pass 2 (Adaptive)
python train.py -s <작업_디렉토리>/pass2
```

2. **품질 비교**:
```bash
# Rendering
python render.py -m <모델_디렉토리>

# Metrics
python metrics.py -m <모델_디렉토리>
```

3. **결과 분석**:
- PSNR, SSIM, LPIPS 비교
- 렌더링 시간 비교
- Visual quality 확인

---

**작성일**: 2026-01-03
**버전**: 1.0
**문의**: PROJECT_SUMMARY.md, README_ADAPTIVE_SAMPLING.md 참고
