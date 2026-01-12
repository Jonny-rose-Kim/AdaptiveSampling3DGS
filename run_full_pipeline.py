#!/usr/bin/env python3
"""
전체 Adaptive Sampling 파이프라인 실행

사용법:
    python run_full_pipeline.py <비디오> <출력_디렉토리> <최종_프레임수> [pass1_fps]

예시:
    python run_full_pipeline.py Museum.mp4 ./output 1000 10
"""

import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    final_frames = int(sys.argv[3])
    pass1_fps = float(sys.argv[4]) if len(sys.argv) > 4 else None

    print("\n" + "="*70)
    print("🚀 Adaptive Sampling 전체 파이프라인")
    print("="*70)

    # Step 1: Pass 1
    print("\n[Step 1/3] Pass 1: 프레임 추출")
    print("-" * 70)

    cmd = ["python", "run_pass1.py", video_path, output_dir]
    if pass1_fps:
        cmd.append(str(pass1_fps))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n❌ Pass 1 실패")
        sys.exit(1)

    # Step 2: COLMAP
    print("\n[Step 2/3] COLMAP 실행")
    print("-" * 70)

    colmap_cmd = ["python", "convert.py", "-s", f"{output_dir}/pass1"]

    print(f"\nCOLMAP 명령어:")
    print(f"  {' '.join(colmap_cmd)}")
    print(f"\n자동 실행하시겠습니까? (y/n): ", end='')

    response = input().strip().lower()
    if response == 'y':
        print("\nCOLMAP 실행 중... (30분~2시간 소요)")
        result = subprocess.run(colmap_cmd)
        if result.returncode != 0:
            print("\n❌ COLMAP 실패")
            sys.exit(1)
    else:
        print("\n⏸️  COLMAP을 수동으로 실행한 후 다음 명령어를 실행하세요:")
        print(f"\n  python run_pass2.py {output_dir} {final_frames}")
        sys.exit(0)

    # Step 3: Pass 2
    print("\n[Step 3/3] Pass 2: Adaptive Sampling")
    print("-" * 70)

    result = subprocess.run(["python", "run_pass2.py", output_dir, str(final_frames)])
    if result.returncode != 0:
        print("\n❌ Pass 2 실패")
        sys.exit(1)

    print("\n" + "="*70)
    print("🎉 전체 파이프라인 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
