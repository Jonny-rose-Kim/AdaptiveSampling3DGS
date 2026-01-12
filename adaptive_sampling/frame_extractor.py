"""
Frame Extractor

ffmpeg를 사용하여 비디오에서 프레임을 추출하는 래퍼 클래스
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """ffmpeg 래퍼 클래스"""

    def __init__(self, video_path: str):
        """
        Args:
            video_path: 비디오 파일 경로
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

    def get_video_info(self) -> dict:
        """
        비디오 정보를 ffprobe로 추출합니다.

        Returns:
            비디오 정보 딕셔너리 (fps, duration, width, height 등)
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'v:0',
            str(self.video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            if 'streams' not in data or len(data['streams']) == 0:
                raise RuntimeError("No video stream found")

            stream = data['streams'][0]

            # FPS 계산
            if 'r_frame_rate' in stream:
                num, den = map(int, stream['r_frame_rate'].split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = 30.0

            # Duration
            duration = float(stream.get('duration', 0))

            # Resolution
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))

            return {
                'fps': fps,
                'duration': duration,
                'width': width,
                'height': height,
                'codec': stream.get('codec_name', 'unknown')
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe output: {e}")
            raise

    def extract_frames_by_fps(
        self,
        output_dir: str,
        fps: Optional[float] = None,
        quality: int = 2
    ) -> List[str]:
        """
        고정 fps로 프레임을 추출합니다.

        Args:
            output_dir: 출력 디렉토리
            fps: 추출할 fps (None이면 원본 fps 사용)
            quality: JPEG 품질 (2=최고 품질, 31=최저 품질)

        Returns:
            추출된 파일 경로 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_pattern = str(output_path / "frame_%06d.png")

        cmd = ['ffmpeg', '-i', str(self.video_path)]

        if fps is not None:
            cmd.extend(['-vf', f'fps={fps}'])

        cmd.extend([
            '-q:v', str(quality),
            output_pattern
        ])

        logger.info(f"Extracting frames with fps={fps if fps else 'original'}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Frame extraction completed")

            # 생성된 파일 리스트 반환
            extracted_files = sorted(output_path.glob("frame_*.png"))
            return [str(f) for f in extracted_files]

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr}")
            raise

    def extract_frames_by_timestamps(
        self,
        timestamps: List[float],
        output_dir: str,
        quality: int = 2,
        accurate_seek: bool = True
    ) -> List[str]:
        """
        특정 timestamp들에서 프레임을 추출합니다.

        Args:
            timestamps: 추출할 timestamp 리스트 (초 단위)
            output_dir: 출력 디렉토리
            quality: JPEG 품질
            accurate_seek: 정확한 seeking 사용 (느리지만 정확)

        Returns:
            추출된 파일 경로 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extracted_files = []

        logger.info(f"Extracting {len(timestamps)} frames by timestamps")

        for i, ts in enumerate(timestamps):
            output_file = output_path / f"frame_{i:06d}_{ts:010.3f}.png"

            cmd = ['ffmpeg']

            if accurate_seek:
                cmd.append('-accurate_seek')

            cmd.extend([
                '-ss', str(ts),
                '-i', str(self.video_path),
                '-frames:v', '1',
                '-q:v', str(quality),
                str(output_file)
            ])

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                extracted_files.append(str(output_file))

                if (i + 1) % 10 == 0:
                    logger.info(f"  Extracted {i + 1}/{len(timestamps)} frames")

            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to extract frame at {ts}s: {e.stderr}")
                continue

        logger.info(f"Extraction completed: {len(extracted_files)}/{len(timestamps)} successful")

        return extracted_files

    def extract_frames_with_metadata(
        self,
        output_dir: str,
        fps: Optional[float] = None
    ) -> Tuple[List[str], dict]:
        """
        프레임을 추출하고 timestamp 메타데이터를 생성합니다.

        Args:
            output_dir: 출력 디렉토리
            fps: 추출할 fps

        Returns:
            (추출된 파일 리스트, timestamp 매핑 딕셔너리)
        """
        extracted_files = self.extract_frames_by_fps(output_dir, fps)

        # Timestamp 메타데이터 생성
        video_info = self.get_video_info()
        extraction_fps = fps if fps is not None else video_info['fps']

        metadata = {}
        for i, file_path in enumerate(extracted_files):
            timestamp = i / extraction_fps
            filename = Path(file_path).name
            metadata[filename] = {
                'index': i,
                'timestamp': timestamp,
                'path': file_path
            }

        # 메타데이터 저장
        metadata_file = Path(output_dir) / "frame_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_file}")

        return extracted_files, metadata


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python frame_extractor.py <video_path> <output_dir> [fps]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else None

    extractor = FrameExtractor(video_path)

    # 비디오 정보 출력
    info = extractor.get_video_info()
    print(f"\nVideo Info:")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  Codec: {info['codec']}")

    # 프레임 추출
    print(f"\nExtracting frames...")
    files, metadata = extractor.extract_frames_with_metadata(output_dir, fps)

    print(f"\nExtracted {len(files)} frames to {output_dir}")
    print(f"First 5 frames:")
    for f in files[:5]:
        print(f"  {Path(f).name}")
