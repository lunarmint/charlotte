import struct
import subprocess

from pathlib import Path
from typing import BinaryIO

import typer


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


class IVF:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.filename = self.file_path.name
        self.header: IVFHeader = self._parse_header()
        self.fps = self.header.framerate / self.header.timescale
        self.duration_ms = (self.header.frames / self.fps) * 1000
        self.frame_duration_ms = (1.0 / self.fps) * 1000

    def _parse_header(self) -> IVFHeader:
        """Parse IVF file header."""
        with open(self.file_path, "rb") as fp:
            return IVFHeader.from_file(fp)

    def convert_to_mp4(self, output_path: Path) -> str:
        """Convert IVF to MP4 using ffmpeg."""
        mp4_file = output_path / f"{Path(self.filename).stem}.mp4"
        typer.echo(f"Converting {self.filename} to MP4...")

        # Build ffmpeg command
        x265_params = [
            "profile=main10",
            ":cutree=0",
            ":deblock=-1,-1",
            ":no-sao=1",
            ":tskip=1",
            ":cbqpoffs=-2",
            ":qcomp=0.7",
            ":lookahead-slices=0",
            ":keyint=300",
            ":min-keyint=30",
            ":max-merge=5",
            ":ref=6",
            ":bframes=16",
            ":rd=4",
            ":psy-rd=2.0",
            ":psy-rdoq=1.5",
            ":aq-mode=3",
            ":aq-strength=0.8",
            ":colorprim=1",
            ":colormatrix=1",
            ":transfer=1",
        ]
        x265_params = "".join(x265_params)
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-loglevel", "error",  # Only show errors
            "-i",
            self.file_path,
            "-c:v", "libx265",
            "-pix_fmt", "yuv420p10le",
            "-vf", "scale=out_color_matrix=bt709",
            "-crf", "12",
            "-preset", "slower",
            "-x265-params",
            x265_params,
            str(mp4_file),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return str(mp4_file)
        except subprocess.CalledProcessError as e:
            typer.echo(f"Error converting video: {e}")
            if e.stderr:
                typer.echo(f"{e.stderr}")
            raise typer.Exit(1) from e
        except FileNotFoundError:
            typer.echo(
                "ffmpeg not found. Place ffmpeg in the root directory and try again."
            )
            raise typer.Exit(1) from None
