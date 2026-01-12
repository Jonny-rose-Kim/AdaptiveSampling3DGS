"""
COLMAP Parser 테스트
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaptive_sampling.colmap_parser import COLMAPParser, CameraPose


def create_test_colmap_data(test_dir: Path):
    """테스트용 COLMAP images.txt 파일 생성"""
    sparse_dir = test_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Sample images.txt content
    images_content = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 3, mean observations per image: 100
1 0.999048 0.0 0.0436194 0.0 1.5 0.0 0.5 1 frame_000000_000.000.png
100.5 200.3 1 150.2 180.7 2
2 0.998027 0.0 0.0627905 0.0 1.8 0.0 0.6 1 frame_000030_001.000.png
105.3 205.1 3 155.8 185.2 4
3 0.996195 0.0 0.0871557 0.0 2.2 0.0 0.8 1 frame_000060_002.000.png
110.7 210.5 5 160.3 190.1 6
"""

    with open(sparse_dir / "images.txt", 'w') as f:
        f.write(images_content)

    # Minimal cameras.txt
    cameras_content = """# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0
"""
    with open(sparse_dir / "cameras.txt", 'w') as f:
        f.write(cameras_content)

    return sparse_dir


def test_parse_images():
    """images.txt 파싱 테스트"""
    print("\n=== Test: Parse Images ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = create_test_colmap_data(Path(tmpdir))

        parser = COLMAPParser(sparse_dir)
        poses = parser.parse_images()

        # 3개의 이미지가 파싱되어야 함
        assert len(poses) == 3, f"Expected 3 poses, got {len(poses)}"
        print(f"✓ Parsed {len(poses)} poses")

        # 첫 번째 이미지 검증
        pose1 = poses[1]
        assert pose1.image_id == 1
        assert pose1.image_name == "frame_000000_000.000.png"
        assert abs(pose1.qw - 0.999048) < 1e-6
        assert abs(pose1.tx - 1.5) < 1e-6
        print(f"✓ First pose: {pose1.image_name}, T={pose1.translation}")

        # Quaternion과 rotation matrix 변환 확인
        quat = pose1.quaternion
        assert quat.shape == (4,)
        print(f"✓ Quaternion: {quat}")

        rot_mat = pose1.rotation_matrix
        assert rot_mat.shape == (3, 3)
        print(f"✓ Rotation matrix shape: {rot_mat.shape}")

        # Rotation matrix가 orthogonal인지 확인
        should_be_identity = rot_mat @ rot_mat.T
        is_orthogonal = np.allclose(should_be_identity, np.eye(3), atol=1e-5)
        assert is_orthogonal, "Rotation matrix is not orthogonal"
        print(f"✓ Rotation matrix is orthogonal")

    print("✓ All parse_images tests passed!")


def test_extract_timestamp():
    """Timestamp 추출 테스트"""
    print("\n=== Test: Extract Timestamp ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = create_test_colmap_data(Path(tmpdir))
        parser = COLMAPParser(sparse_dir)

        # 명시적 timestamp가 있는 경우
        ts1 = parser.extract_timestamp_from_filename("frame_000000_000.000.png")
        assert ts1 == 0.0, f"Expected 0.0, got {ts1}"
        print(f"✓ Explicit timestamp: frame_000000_000.000.png -> {ts1}s")

        ts2 = parser.extract_timestamp_from_filename("frame_000030_001.000.png")
        assert ts2 == 1.0, f"Expected 1.0, got {ts2}"
        print(f"✓ Explicit timestamp: frame_000030_001.000.png -> {ts2}s")

        # fps 기반 계산
        ts3 = parser.extract_timestamp_from_filename("frame_000060.png", fps=30.0)
        assert ts3 == 2.0, f"Expected 2.0, got {ts3}"
        print(f"✓ FPS-based timestamp: frame_000060.png @ 30fps -> {ts3}s")

        # 간단한 숫자 파일명
        ts4 = parser.extract_timestamp_from_filename("000090.png", fps=30.0)
        assert ts4 == 3.0, f"Expected 3.0, got {ts4}"
        print(f"✓ FPS-based timestamp: 000090.png @ 30fps -> {ts4}s")

    print("✓ All extract_timestamp tests passed!")


def test_add_timestamps_and_sort():
    """Timestamp 추가 및 정렬 테스트"""
    print("\n=== Test: Add Timestamps and Sort ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = create_test_colmap_data(Path(tmpdir))
        parser = COLMAPParser(sparse_dir)

        poses = parser.parse_images()
        parser.add_timestamps(poses)

        # 모든 pose에 timestamp가 추가되었는지 확인
        for pose in poses.values():
            assert pose.timestamp is not None, f"Timestamp not set for {pose.image_name}"
        print(f"✓ All {len(poses)} poses have timestamps")

        # 정렬된 리스트 확인
        sorted_poses = parser.get_sorted_poses(poses)
        assert len(sorted_poses) == 3
        print(f"✓ Sorted poses: {len(sorted_poses)}")

        # Timestamp가 오름차순인지 확인
        timestamps = [p.timestamp for p in sorted_poses]
        assert timestamps == sorted(timestamps), "Timestamps are not sorted"
        print(f"✓ Timestamps are sorted: {timestamps}")

        # 각 pose 정보 출력
        for i, pose in enumerate(sorted_poses):
            print(f"  [{i}] {pose.image_name}: t={pose.timestamp:.3f}s, T={pose.translation}")

    print("✓ All add_timestamps_and_sort tests passed!")


def test_parse_and_extract():
    """전체 파이프라인 테스트"""
    print("\n=== Test: Full Pipeline (parse_and_extract) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = create_test_colmap_data(Path(tmpdir))
        parser = COLMAPParser(sparse_dir)

        # 전체 파이프라인 실행
        sorted_poses = parser.parse_and_extract()

        assert len(sorted_poses) == 3
        print(f"✓ Extracted {len(sorted_poses)} sorted poses")

        # 첫 번째와 마지막 확인
        first = sorted_poses[0]
        last = sorted_poses[-1]
        print(f"✓ First: {first.image_name} @ {first.timestamp:.3f}s")
        print(f"✓ Last: {last.image_name} @ {last.timestamp:.3f}s")

        assert first.timestamp <= last.timestamp, "First timestamp should be <= last"
        print(f"✓ Timestamps are properly ordered")

    print("✓ All parse_and_extract tests passed!")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("Running COLMAP Parser Tests")
    print("="*60)

    try:
        test_parse_images()
        test_extract_timestamp()
        test_add_timestamps_and_sort()
        test_parse_and_extract()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
