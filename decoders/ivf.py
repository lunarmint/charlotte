"""IVF (VP9) video format decoder.

This module parses IVF container files containing VP9 video frames.
IVF is a simple container format commonly used for VP9 video codec.
"""

import struct
import subprocess

from pathlib import Path
from typing import BinaryIO


# IVF format constants
IVF_SIGNATURE = 0x46494B44  # "DKIF" in little-endian
HEADER_MIN_SIZE = 28
FRAME_HEADER_SIZE = 12


class IVFHeader:
    """IVF file header structure."""

    __slots__ = (
        "codec",
        "framerate",
        "frames",
        "header_length",
        "height",
        "timescale",
        "version",
        "width",
    )

    def __init__(self):
        self.version = 0
        self.header_length = 0
        self.codec = ""
        self.width = 0
        self.height = 0
        self.framerate = 0
        self.timescale = 0
        self.frames = 0

    @classmethod
    def from_file(cls, fp: BinaryIO) -> "IVFHeader":
        """Parse IVF header from file."""
        header = cls()

        # Verify signature
        signature = struct.unpack("<I", fp.read(4))[0]
        if signature != IVF_SIGNATURE:
            raise ValueError(f"Invalid IVF file: wrong signature 0x{signature:08X}")

        # Read header fields
        header.version = struct.unpack("<H", fp.read(2))[0]
        header.header_length = struct.unpack("<H", fp.read(2))[0]
        header.codec = fp.read(4).decode("ascii")
        header.width = struct.unpack("<H", fp.read(2))[0]
        header.height = struct.unpack("<H", fp.read(2))[0]
        header.framerate = struct.unpack("<I", fp.read(4))[0]
        header.timescale = struct.unpack("<I", fp.read(4))[0]
        header.frames = struct.unpack("<I", fp.read(4))[0]

        # Skip unused padding bytes
        fp.read(header.header_length - HEADER_MIN_SIZE)

        return header


class IVFFrame:
    """IVF video frame."""

    __slots__ = ("data", "length", "timestamp")

    def __init__(self, data: bytes, length: int, timestamp: int):
        self.data = data
        self.length = length
        self.timestamp = timestamp


class IVF:
    """IVF container parser for VP9 video.

    Parses IVF files and extracts VP9 video frames with metadata.

    Args:
        file_path: Path to the IVF file
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.header: IVFHeader = self._parse_header()
        self.fps = self.header.framerate / self.header.timescale
        self.duration_ms = (self.header.frames / self.fps) * 1000
        self.frame_duration_ms = (1.0 / self.fps) * 1000

    def _parse_header(self) -> IVFHeader:
        """Parse IVF file header."""
        with open(self.file_path, "rb") as fp:
            return IVFHeader.from_file(fp)

    def get_info(self) -> dict[str, any]:
        """Get video information.

        Returns:
            Dictionary with video metadata
        """
        return {
            "codec": self.header.codec,
            "width": self.header.width,
            "height": self.header.height,
            "fps": self.fps,
            "frames": self.header.frames,
            "duration_ms": self.duration_ms,
        }

    def read_frames(self) -> list[IVFFrame]:
        """Read all video frames.

        Returns:
            List of IVFFrame objects
        """
        frames = []
        with open(self.file_path, "rb") as fp:
            fp.seek(self.header.header_length)

            for _ in range(self.header.frames):
                # Read frame header
                frame_data = fp.read(FRAME_HEADER_SIZE)
                if len(frame_data) < FRAME_HEADER_SIZE:
                    break

                length = struct.unpack("<I", frame_data[:4])[0]
                timestamp = struct.unpack("<Q", frame_data[4:12])[0]

                # Read frame data
                data = fp.read(length)
                if len(data) < length:
                    break

                frames.append(IVFFrame(length, timestamp, data))

        return frames

    def extract_frames(self, output_dir: str = ".") -> list[str]:
        """Extract all frames as raw VP9 data files.

        Args:
            output_dir: Output directory for frame files

        Returns:
            List of frame file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        base_name = self.file_path.stem

        print(f"Extracting frames from {self.file_path.name}...")
        print(f"  Resolution: {self.header.width}x{self.header.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Frames: {self.header.frames}")

        frame_files = []
        frames = self.read_frames()

        for i, frame in enumerate(frames, 1):
            frame_path = output_path / f"{base_name}_frame_{i:06d}.vp9"
            frame_path.write_bytes(frame.data)
            frame_files.append(str(frame_path))

            if i % 100 == 0:
                print(f"  Extracted {i}/{self.header.frames} frames...")

        print(f"Extraction complete: {len(frame_files)} frames")
        return frame_files

    def convert_to_mp4(
        self,
        crf: int,
        preset: str,
        output_path: str | None = None,
        codec: str = "copy",
        pixel_format: str = "yuv420p10le",
    ) -> str:
        if output_path is None:
            output_path = str(self.file_path.with_suffix(".mp4"))

        print(f"Converting {self.file_path.name} to MP4...")
        print(f"  Resolution: {self.header.width}x{self.header.height}")
        print(f"  FPS: {self.fps}")
        if codec != "copy":
            print(f"  Codec: {codec}, CRF: {crf}, Preset: {preset}")

        # Find ffmpeg executable
        ffmpeg_cmd = self._find_ffmpeg()

        # Build ffmpeg command
        cmd = [ffmpeg_cmd, "-i", str(self.file_path)]

        if codec == "copy":
            cmd.extend(["-c:v", "copy"])
        else:
            cmd.extend(
                [
                    "-c:v",
                    codec,
                    "-crf",
                    str(crf),
                    "-preset",
                    preset,
                    "-pix_fmt",
                    pixel_format,
                ]
            )

        cmd.extend(["-y", output_path])

        # Execute conversion
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Conversion complete: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error converting video: {e}")
            print(f"stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            raise FileNotFoundError(
                "ffmpeg not found. Please install ffmpeg to convert videos."
            )

    @staticmethod
    def _find_ffmpeg() -> str:
        """Find ffmpeg executable in project root or PATH."""
        project_ffmpeg = Path("ffmpeg.exe")
        if project_ffmpeg.exists():
            return str(project_ffmpeg.absolute())
        return "ffmpeg"
