#!/usr/bin/env python3
"""
Pass 1: 비디오에서 고정 fps로 프레임 추출

사용법:
    python run_pass1.py <비디오_경로> <출력_디렉토리> [fps]

예시:
    python run_pass1.py Museum.mp4 ./museum_output 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptive_sampling.frame_extractor import FrameExtractor
import json


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else None

    print("\n" + "="*70)
    print("Pass 1: 초기 프레임 추출")
    print("="*70)

    # 비디오 정보
    extractor = FrameExtractor(video_path)
    info = extractor.get_video_info()

    print(f"\n비디오 정보:")
    print(f"  파일: {video_path}")
    print(f"  해상도: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  길이: {info['duration']:.2f}초")
    print(f"  총 프레임: {int(info['duration'] * info['fps'])}개")

    # FPS 설정
    extraction_fps = fps if fps else info['fps']
    expected_frames = int(info['duration'] * extraction_fps)

    print(f"\n추출 설정:")
    print(f"  추출 FPS: {extraction_fps:.2f}")
    print(f"  예상 프레임 수: {expected_frames}개")

    # 프레임 추출
    frames_dir = Path(output_dir) / "pass1" / "input"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n프레임 추출 중...")
    print(f"  출력: {frames_dir}")

    files, metadata = extractor.extract_frames_with_metadata(
        str(frames_dir),
        fps=extraction_fps
    )

    print(f"\n✅ 완료!")
    print(f"  추출된 프레임: {len(files)}개")
    print(f"  출력 디렉토리: {frames_dir}")

    # 설정 저장
    config = {
        'video_path': str(Path(video_path).absolute()),
        'output_dir': str(Path(output_dir).absolute()),
        'extraction_fps': extraction_fps,
        'video_info': info,
        'extracted_frames': len(files)
    }

    config_file = Path(output_dir) / "pass1_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  설정 저장: {config_file}")

    # 다음 단계 안내
    print(f"\n" + "="*70)
    print("다음 단계: COLMAP 실행")
    print("="*70)
    print(f"\n방법 1: convert.py 사용 (권장)")
    print(f"  python convert.py -s {output_dir}/pass1")

    print(f"\n방법 2: COLMAP 수동 실행")
    print(f"  python run_colmap.py {output_dir}")

    print(f"\n진행 상황 확인:")
    print(f"  nvidia-smi  # GPU 사용 확인")
    print(f"  ps aux | grep colmap  # 프로세스 확인")


if __name__ == "__main__":
    main()
