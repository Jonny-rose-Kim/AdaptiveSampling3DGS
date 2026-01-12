#!/usr/bin/env python3
"""
TensorBoard 로그에서 PSNR과 Loss를 추출하여 비교하는 스크립트
"""

import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import json


def extract_metrics(logdir: Path, name: str):
    """TensorBoard event 파일에서 metrics 추출"""
    print(f"\n{'='*60}")
    print(f"Extracting metrics from: {name}")
    print(f"{'='*60}")

    # event 파일 찾기
    event_files = list(logdir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"❌ No event files found in {logdir}")
        return None

    event_file = event_files[0]
    print(f"Reading: {event_file.name}")

    # TensorBoard event accumulator
    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={
            event_accumulator.SCALARS: 0,  # 모든 scalar 데이터 로드
        }
    )
    ea.Reload()

    # 사용 가능한 tags 확인
    available_tags = ea.Tags()['scalars']
    print(f"\nAvailable metrics: {', '.join(available_tags)}")

    # Metrics 추출
    metrics = {}

    # Loss metrics
    loss_tags = ['train_loss_patches/l1_loss', 'train_loss_patches/total_loss',
                 'Loss', 'loss', 'total_loss', 'train/loss_viewpoint - l1_loss']
    for tag in loss_tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            df = pd.DataFrame(events)
            metrics[f'loss_{tag}'] = {
                'final': df.iloc[-1]['value'] if len(df) > 0 else None,
                'mean': df['value'].mean(),
                'min': df['value'].min(),
                'data': df[['step', 'value']].to_dict('records')
            }

    # PSNR metrics
    psnr_tags = ['train_psnr', 'test_psnr', 'PSNR', 'psnr', 'train/loss_viewpoint - psnr']
    for tag in psnr_tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            df = pd.DataFrame(events)
            metrics[f'psnr_{tag}'] = {
                'final': df.iloc[-1]['value'] if len(df) > 0 else None,
                'mean': df['value'].mean(),
                'max': df['value'].max(),
                'data': df[['step', 'value']].to_dict('records')
            }

    return metrics


def print_comparison(adaptive_metrics, original_metrics):
    """두 모델의 metrics 비교 출력"""
    print(f"\n{'='*60}")
    print("📊 METRICS COMPARISON")
    print(f"{'='*60}\n")

    # Loss 비교
    print("📉 LOSS Metrics:")
    print("-" * 60)
    for key in adaptive_metrics.keys():
        if key.startswith('loss_'):
            adaptive_val = adaptive_metrics[key]
            original_val = original_metrics.get(key, {})

            metric_name = key.replace('loss_', '')
            print(f"\n{metric_name}:")
            print(f"  Adaptive  - Final: {adaptive_val.get('final', 'N/A'):.6f}, "
                  f"Mean: {adaptive_val.get('mean', 'N/A'):.6f}, "
                  f"Min: {adaptive_val.get('min', 'N/A'):.6f}")
            print(f"  Original  - Final: {original_val.get('final', 'N/A'):.6f}, "
                  f"Mean: {original_val.get('mean', 'N/A'):.6f}, "
                  f"Min: {original_val.get('min', 'N/A'):.6f}")

            if adaptive_val.get('final') and original_val.get('final'):
                diff = adaptive_val['final'] - original_val['final']
                pct = (diff / original_val['final']) * 100
                symbol = "✅" if diff < 0 else "❌"
                print(f"  {symbol} Difference: {diff:+.6f} ({pct:+.2f}%)")

    # PSNR 비교
    print(f"\n{'='*60}")
    print("📈 PSNR Metrics:")
    print("-" * 60)
    for key in adaptive_metrics.keys():
        if key.startswith('psnr_'):
            adaptive_val = adaptive_metrics[key]
            original_val = original_metrics.get(key, {})

            metric_name = key.replace('psnr_', '')
            print(f"\n{metric_name}:")
            print(f"  Adaptive  - Final: {adaptive_val.get('final', 'N/A'):.4f}, "
                  f"Mean: {adaptive_val.get('mean', 'N/A'):.4f}, "
                  f"Max: {adaptive_val.get('max', 'N/A'):.4f}")
            print(f"  Original  - Final: {original_val.get('final', 'N/A'):.4f}, "
                  f"Mean: {original_val.get('mean', 'N/A'):.4f}, "
                  f"Max: {original_val.get('max', 'N/A'):.4f}")

            if adaptive_val.get('final') and original_val.get('final'):
                diff = adaptive_val['final'] - original_val['final']
                pct = (diff / original_val['final']) * 100
                symbol = "✅" if diff > 0 else "❌"
                print(f"  {symbol} Difference: {diff:+.4f} ({pct:+.2f}%)")


def main():
    # 경로 설정 (스크립트 위치 기준)
    script_dir = Path(__file__).parent.parent
    adaptive_dir = script_dir / "output/museum_adaptive/museum_adaptive"
    original_dir = script_dir / "output/museum_original/museum_original"

    # Metrics 추출
    adaptive_metrics = extract_metrics(adaptive_dir, "Adaptive Sampling")
    original_metrics = extract_metrics(original_dir, "Original Sampling")

    if adaptive_metrics is None or original_metrics is None:
        print("❌ Failed to extract metrics")
        return

    # 비교 출력
    print_comparison(adaptive_metrics, original_metrics)

    # JSON으로 저장
    output_file = Path("metrics_comparison.json")
    with open(output_file, 'w') as f:
        json.dump({
            'adaptive': adaptive_metrics,
            'original': original_metrics
        }, f, indent=2)
    print(f"\n✅ Detailed metrics saved to: {output_file}")


if __name__ == "__main__":
    main()
