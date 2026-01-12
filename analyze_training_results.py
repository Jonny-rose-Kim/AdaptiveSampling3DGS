#!/usr/bin/env python3
"""
3DGS 학습 결과 분석: Baseline vs Adaptive
"""

print("=" * 80)
print("3DGS 학습 결과 분석: Museum_cut")
print("=" * 80)

# Training results
results = {
    'Pass 1 (Baseline)': {
        'initial_points': 182997,
        'iter_7000': {'L1': 0.0280, 'PSNR': 26.98},
        'iter_30000': {'L1': 0.0179, 'PSNR': 30.82},
    },
    'Pass 2 (Adaptive)': {
        'initial_points': 178896,
        'iter_7000': {'L1': 0.0296, 'PSNR': 27.17},
        'iter_30000': {'L1': 0.0199, 'PSNR': 30.60},
    }
}

print("\n" + "=" * 80)
print("학습 메트릭 비교")
print("=" * 80)

print(f"\n초기 상태:")
print(f"  Baseline: {results['Pass 1 (Baseline)']['initial_points']:,} points")
print(f"  Adaptive: {results['Pass 2 (Adaptive)']['initial_points']:,} points")
diff_init = results['Pass 1 (Baseline)']['initial_points'] - results['Pass 2 (Adaptive)']['initial_points']
print(f"  차이: {diff_init:+,} points ({diff_init/results['Pass 2 (Adaptive)']['initial_points']*100:+.2f}%)")

print(f"\nITER 7000 (Early Stage):")
base_7k = results['Pass 1 (Baseline)']['iter_7000']
adapt_7k = results['Pass 2 (Adaptive)']['iter_7000']
print(f"  Baseline: L1={base_7k['L1']:.6f}, PSNR={base_7k['PSNR']:.2f}")
print(f"  Adaptive: L1={adapt_7k['L1']:.6f}, PSNR={adapt_7k['PSNR']:.2f}")
print(f"  PSNR 차이: {adapt_7k['PSNR'] - base_7k['PSNR']:+.2f} dB")
if adapt_7k['PSNR'] > base_7k['PSNR']:
    print(f"  ✓ Adaptive가 {adapt_7k['PSNR'] - base_7k['PSNR']:.2f} dB 우세!")

print(f"\nITER 30000 (Final):")
base_30k = results['Pass 1 (Baseline)']['iter_30000']
adapt_30k = results['Pass 2 (Adaptive)']['iter_30000']
print(f"  Baseline: L1={base_30k['L1']:.6f}, PSNR={base_30k['PSNR']:.2f}")
print(f"  Adaptive: L1={adapt_30k['L1']:.6f}, PSNR={adapt_30k['PSNR']:.2f}")
print(f"  PSNR 차이: {adapt_30k['PSNR'] - base_30k['PSNR']:+.2f} dB")
if base_30k['PSNR'] > adapt_30k['PSNR']:
    print(f"  ⚠️  Baseline이 {base_30k['PSNR'] - adapt_30k['PSNR']:.2f} dB 역전!")

print("\n" + "=" * 80)
print("학습 진행 분석")
print("=" * 80)

# Improvement over training
base_improvement = base_30k['PSNR'] - base_7k['PSNR']
adapt_improvement = adapt_30k['PSNR'] - adapt_7k['PSNR']

print(f"\n7000 → 30000 iteration 개선량:")
print(f"  Baseline: {base_improvement:+.2f} dB")
print(f"  Adaptive: {adapt_improvement:+.2f} dB")
print(f"  차이: {base_improvement - adapt_improvement:+.2f} dB")

if base_improvement > adapt_improvement:
    print(f"\n⚠️  Baseline이 {base_improvement - adapt_improvement:.2f} dB 더 많이 개선됨")
    print(f"   → Adaptive는 초기 수렴은 빠르지만, 장기 개선이 느림")

# Learning efficiency
base_lr_7k = base_improvement / (30000 - 7000)
adapt_lr_7k = adapt_improvement / (30000 - 7000)

print(f"\n후반 학습 효율 (7000~30000):")
print(f"  Baseline: {base_lr_7k*10000:.4f} dB/10k iter")
print(f"  Adaptive: {adapt_lr_7k*10000:.4f} dB/10k iter")

print("\n" + "=" * 80)
print("원인 분석")
print("=" * 80)

print(f"""
관찰된 패턴:
1. 초기 (7000 iter): Adaptive > Baseline ({adapt_7k['PSNR']:.2f} vs {base_7k['PSNR']:.2f})
2. 최종 (30000 iter): Baseline > Adaptive ({base_30k['PSNR']:.2f} vs {adapt_30k['PSNR']:.2f})
3. 후반 개선량: Baseline이 {base_improvement - adapt_improvement:.2f} dB 더 큼

가능한 원인:

┌──────────────────────────────────────────────────────────────────┐
│ 원인 1: 초기 Points 수 차이                                      │
├──────────────────────────────────────────────────────────────────┤
│ Baseline: {results['Pass 1 (Baseline)']['initial_points']:,} points (SfM 재구성)                          │
│ Adaptive: {results['Pass 2 (Adaptive)']['initial_points']:,} points                                       │
│ 차이: {diff_init:,} points ({diff_init/results['Pass 2 (Adaptive)']['initial_points']*100:.2f}%)                                            │
│                                                                  │
│ 영향:                                                            │
│ - 더 많은 초기 points = 더 좋은 geometry 초기화                 │
│ - Densification 과정에서 더 많은 "seed"                         │
│ - 최종적으로 더 세밀한 표현 가능                                │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ 원인 2: Feature Coverage 차이                                    │
├──────────────────────────────────────────────────────────────────┤
│ Adaptive의 frame 분포:                                           │
│ - CV = 0.352 (변동성 있음)                                       │
│ - 빠른 구간: 간격 좁음 (redundant features?)                    │
│ - 느린 구간: 간격 넓음 (coverage gap?)                          │
│                                                                  │
│ Baseline의 frame 분포:                                           │
│ - CV = 0.0 (완전 균등)                                           │
│ - 일정한 간격 = 균일한 coverage                                  │
│                                                                  │
│ 가설:                                                            │
│ Adaptive의 "gap" 영역에서 feature matching 실패                 │
│ → SfM points 감소 ({diff_init:,} points)                                  │
│ → 최종 품질 저하                                                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ 원인 3: Densification 차이                                       │
├──────────────────────────────────────────────────────────────────┤
│ 3DGS Densification은 gradient 기반:                              │
│ - 높은 gradient 영역에 Gaussian 추가                            │
│ - 초기 points가 많을수록 → 더 정확한 gradient 계산              │
│                                                                  │
│ Baseline: 182,997 points로 시작                                  │
│ → 풍부한 초기 정보                                               │
│ → Densification이 올바른 위치에                                  │
│                                                                  │
│ Adaptive: 178,896 points로 시작                                  │
│ → 일부 영역 정보 부족                                            │
│ → Densification이 차선책 위치에                                  │
│ → 장기적으로 품질 저하                                           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ 원인 4: Early vs Late Stage 최적화                              │
├──────────────────────────────────────────────────────────────────┤
│ Early Stage (0~7000):                                            │
│ - 주로 opacity, position 최적화                                  │
│ - Adaptive의 밀집된 frame이 유리 (빠른 수렴)                    │
│                                                                  │
│ Late Stage (7000~30000):                                         │
│ - Fine-tuning, detail 최적화                                     │
│ - 균일한 coverage가 유리 (모든 영역 개선)                       │
│ - Baseline의 균등 분포가 승리                                    │
└──────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("검증 방법")
print("=" * 80)

print(f"""
다음 정보를 확인해야 합니다:

1. 최종 Gaussian 개수:
   ls -lh output/pass1/point_cloud/iteration_30000/point_cloud.ply
   ls -lh output/pass2/point_cloud/iteration_30000/point_cloud.ply
   
2. Densification 과정 로그:
   - Baseline과 Adaptive의 Gaussian 개수 변화 추이
   - Densification/Pruning 비율
   
3. Test set 성능:
   python render.py -m output/pass1
   python render.py -m output/pass2
   python metrics.py -m output/pass1
   python metrics.py -m output/pass2
   
4. Visual inspection:
   - 어느 영역에서 차이가 나는지
   - Adaptive의 sparse 구간에서 artifact가 있는지
""")

print("\n" + "=" * 80)
print("결론 및 권장사항")
print("=" * 80)

print(f"""
현재 결과 해석:

✓ Adaptive sampling의 장점:
  - 초기 수렴 빠름 (7000 iter에서 {adapt_7k['PSNR'] - base_7k['PSNR']:+.2f} dB 우세)
  - 초기 학습 단계에서 효율적

⚠️  Adaptive sampling의 단점:
  - 최종 품질 낮음 (30000 iter에서 {adapt_30k['PSNR'] - base_30k['PSNR']:+.2f} dB 열세)
  - SfM points {diff_init} 개 부족
  - 장기 학습 효율 낮음

근본 원인 (추정):
  Museum scene의 adaptive 분포가 오히려 feature coverage에 gap 발생
  → SfM reconstruction 품질 저하
  → 3DGS 학습의 "천장" 낮아짐

권장사항:

1. 단기 (현재 Museum scene):
   → Baseline 사용 (PSNR 30.82 > 30.60)
   
2. 중기 (검증):
   → Test set 성능 확인
   → Visual quality 비교
   → High-motion scene 테스트
   
3. 장기 (개선):
   → Adaptive sampling 파라미터 조정
   → Minimum interval constraint 추가
   → Hybrid approach (80% uniform + 20% adaptive)
""")

print("=" * 80)

