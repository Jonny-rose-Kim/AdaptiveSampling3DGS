#!/usr/bin/env python3
"""
Compare SfM reconstruction results across all Pass 2 versions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from adaptive_sampling.colmap_parser import COLMAPParser
import numpy as np


def count_points3d(filepath):
    """Count 3D points from text file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return sum(1 for line in lines if line.strip() and not line.startswith('#'))


def main():
    base_dir = '/home/jonny/jonny/Adaptive-ffmpeg/data/Museum_cut_exp'

    print("=" * 80)
    print("Complete Comparison: Pass 1 vs Pass 2 (Old) vs Pass 2 (New)")
    print("=" * 80)

    results = {}

    # Pass 1
    pass1_dir = os.path.join(base_dir, 'pass1/sparse/0')
    if os.path.exists(pass1_dir):
        parser1 = COLMAPParser(pass1_dir)
        poses1 = parser1.parse_images()
        points1_file = os.path.join(pass1_dir, 'points3D.txt')
        if os.path.exists(points1_file):
            points1 = count_points3d(points1_file)
        else:
            points1 = 0

        results['Pass 1 (Uniform)'] = {
            'cameras': len(poses1),
            'points': points1,
            'description': 'Baseline - uniform temporal sampling (0.5s)'
        }

    # Pass 2 Old
    pass2_old_dir = os.path.join(base_dir, 'pass2_old/sparse/0')
    if os.path.exists(pass2_old_dir):
        parser2_old = COLMAPParser(pass2_old_dir)
        poses2_old = parser2_old.parse_images()
        points2_old_file = os.path.join(pass2_old_dir, 'points3D.txt')
        if os.path.exists(points2_old_file):
            points2_old = count_points3d(points2_old_file)
        else:
            points2_old = 0

        results['Pass 2 Old (Max Norm)'] = {
            'cameras': len(poses2_old),
            'points': points2_old,
            'description': 'Original - max normalization + broken sparse handling'
        }

    # Pass 2 New
    pass2_new_dir = os.path.join(base_dir, 'pass2/sparse/0')
    if os.path.exists(pass2_new_dir):
        # Wait for COLMAP to finish
        if not os.path.exists(os.path.join(pass2_new_dir, 'images.bin')):
            print("\n⏳ Waiting for COLMAP to complete on Pass 2 (New)...")
            return

        # Convert binary to text if needed
        if not os.path.exists(os.path.join(pass2_new_dir, 'points3D.txt')):
            import subprocess
            subprocess.run([
                'colmap', 'model_converter',
                '--input_path', pass2_new_dir,
                '--output_path', pass2_new_dir,
                '--output_type', 'TXT'
            ], capture_output=True)

        parser2_new = COLMAPParser(pass2_new_dir)
        poses2_new = parser2_new.parse_images()
        points2_new_file = os.path.join(pass2_new_dir, 'points3D.txt')
        if os.path.exists(points2_new_file):
            points2_new = count_points3d(points2_new_file)
        else:
            points2_new = 0

        results['Pass 2 New (Percentile)'] = {
            'cameras': len(poses2_new),
            'points': points2_new,
            'description': 'Fixed - 95th percentile norm + score-based merging'
        }

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for name, data in results.items():
        print(f"\n{name}")
        print(f"  {data['description']}")
        print(f"  Cameras: {data['cameras']}")
        print(f"  3D Points: {data['points']:,}")

    # Calculate improvements
    if 'Pass 1 (Uniform)' in results:
        baseline_points = results['Pass 1 (Uniform)']['points']

        print("\n" + "=" * 80)
        print("IMPROVEMENT vs BASELINE")
        print("=" * 80)

        for name, data in results.items():
            if name == 'Pass 1 (Uniform)':
                continue

            diff = data['points'] - baseline_points
            pct = (diff / baseline_points * 100) if baseline_points > 0 else 0

            print(f"\n{name}:")
            print(f"  Difference: {diff:+,} points ({pct:+.2f}%)")

            if pct > 2:
                print(f"  ✅ SIGNIFICANT IMPROVEMENT")
            elif pct > 0:
                print(f"  ✓ Slight improvement")
            elif pct > -2:
                print(f"  → Similar performance")
            else:
                print(f"  ⚠️  Degradation")

    # Analyze frame distribution differences
    print("\n" + "=" * 80)
    print("FRAME DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Pass 2 Old timestamps
    old_ts_file = os.path.join(base_dir, 'pass2_old_timestamps.json') if os.path.exists(os.path.join(base_dir, 'pass2_old_timestamps.json')) else None
    # Pass 2 New timestamps
    new_ts_file = os.path.join(base_dir, 'adaptive_timestamps.json')

    if old_ts_file and os.path.exists(new_ts_file):
        import json

        if os.path.exists(old_ts_file):
            with open(old_ts_file, 'r') as f:
                old_data = json.load(f)
            old_timestamps = old_data.get('timestamps', [])
        else:
            old_timestamps = []

        with open(new_ts_file, 'r') as f:
            new_data = json.load(f)
        new_timestamps = new_data.get('timestamps', [])

        if old_timestamps:
            old_intervals = [old_timestamps[i+1] - old_timestamps[i] for i in range(len(old_timestamps)-1)]
            print(f"\nPass 2 Old:")
            print(f"  Mean interval: {np.mean(old_intervals):.3f}s")
            print(f"  Std interval: {np.std(old_intervals):.3f}s")
            print(f"  CV: {np.std(old_intervals)/np.mean(old_intervals):.4f}")

        if new_timestamps:
            new_intervals = [new_timestamps[i+1] - new_timestamps[i] for i in range(len(new_timestamps)-1)]
            print(f"\nPass 2 New:")
            print(f"  Mean interval: {np.mean(new_intervals):.3f}s")
            print(f"  Std interval: {np.std(new_intervals):.3f}s")
            print(f"  CV: {np.std(new_intervals)/np.mean(new_intervals):.4f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if 'Pass 2 New (Percentile)' in results and 'Pass 1 (Uniform)' in results:
        new_points = results['Pass 2 New (Percentile)']['points']
        base_points = results['Pass 1 (Uniform)']['points']
        improvement = ((new_points - base_points) / base_points * 100) if base_points > 0 else 0

        print(f"\nFinal Pass 2 (with fixes) vs Baseline:")
        print(f"  {improvement:+.2f}% change in 3D points")

        if improvement > 5:
            print("\n✅ SUCCESS: Significant improvement achieved!")
            print("  - Percentile normalization improved score distribution")
            print("  - Score-based merging preserved important frames")
        elif improvement > 0:
            print("\n✓ IMPROVEMENT: Positive results")
            print("  - Small gain may indicate scene has limited motion variation")
        elif improvement > -5:
            print("\n→ NEUTRAL: Similar performance")
            print("  - Scene may not benefit from adaptive sampling")
            print("  - Consider testing on high-motion datasets")
        else:
            print("\n⚠️  INVESTIGATION NEEDED")
            print("  - Check parameter tuning (alpha, beta, threshold_multiplier)")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
