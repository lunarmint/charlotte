"""USM (CRI Sofdec) container format demuxer.

This module demuxes CRI Middleware's USM container files, extracting:
- Video streams (VP9 codec in IVF format)
- Audio streams (HCA format)

USM files are commonly used in video games for cutscenes and movies.
"""
import typer
from pathlib import Path
from typing import BinaryIO

from decoders.ivf import IVFWriter, parse_vp9_dimensions


# USM chunk signatures
SIG_CRID = 0x43524944  # CRID - Container ID
SIG_VIDEO = 0x40534656  # @SFV - Video chunk
SIG_AUDIO = 0x40534641  # @SFA - Audio chunk
SIG_CUE = 0x40435545    # @CUE - Cue point

HEADER_SIZE = 32
VIDEO_OFFSET = 0x40
AUDIO_OFFSET = 0x140
MASK_SIZE = 0x20
MIN_VIDEO_SIZE = 0x200


class ChunkHeader:
    """USM chunk header structure."""

    __slots__ = (
        "signature", "data_size", "data_offset", "padding_size",
        "channel_no", "data_type", "frame_time", "frame_rate"
    )

    def __init__(self):
        self.signature = 0
        self.data_size = 0
        self.data_offset = 0
        self.padding_size = 0
        self.channel_no = 0
        self.data_type = 0
        self.frame_time = 0
        self.frame_rate = 0

    @classmethod
    def from_bytes(cls, data: bytes) -> "ChunkHeader":
        """Parse chunk header from 32-byte block."""
        header = cls()
        header.signature = int.from_bytes(data[0:4], "big")
        header.data_size = int.from_bytes(data[4:8], "big")
        header.data_offset = data[9]
        header.padding_size = int.from_bytes(data[10:12], "big")
        header.channel_no = data[12]
        header.data_type = data[15]
        header.frame_time = int.from_bytes(data[16:20], "big")
        header.frame_rate = int.from_bytes(data[20:24], "big")
        return header


class USM:
    """USM container demuxer.

    Demuxes USM files and extracts video (VP9/IVF) and audio (HCA) streams.
    Handles encrypted USM files using provided keys.

    Args:
        file_path: Path to the USM file
        key1: Decryption key (4 bytes) for video/audio
        key2: Decryption key (4 bytes) for video/audio
    """

    def __init__(self, file_path: Path, key1: bytes, key2: bytes):
        self.file_path = Path(file_path)
        self.key1 = key1
        self.key2 = key2

        # Initialize decryption masks
        self.video_mask1 = bytearray(MASK_SIZE)
        self.video_mask2 = bytearray(MASK_SIZE)
        self.audio_mask = bytearray(MASK_SIZE)
        self._init_masks(key1, key2)

    def _init_masks(self, key1: bytes, key2: bytes) -> None:
        """Initialize decryption masks from keys."""
        m = self.video_mask1  # Shorthand for readability

        # Initialize base mask from keys
        m[0x00] = key1[0]
        m[0x01] = key1[1]
        m[0x02] = key1[2]
        m[0x03] = (key1[3] - 0x34) & 0xFF
        m[0x04] = (key2[0] + 0xF9) & 0xFF
        m[0x05] = (key2[1] ^ 0x13) & 0xFF
        m[0x06] = (key2[2] + 0x61) & 0xFF

        # Derive remaining mask values
        m[0x07] = (m[0x00] ^ 0xFF) & 0xFF
        m[0x08] = (m[0x02] + m[0x01]) & 0xFF
        m[0x09] = (m[0x01] - m[0x07]) & 0xFF
        m[0x0A] = (m[0x02] ^ 0xFF) & 0xFF
        m[0x0B] = (m[0x01] ^ 0xFF) & 0xFF
        m[0x0C] = (m[0x0B] + m[0x09]) & 0xFF
        m[0x0D] = (m[0x08] - m[0x03]) & 0xFF
        m[0x0E] = (m[0x0D] ^ 0xFF) & 0xFF
        m[0x0F] = (m[0x0A] - m[0x0B]) & 0xFF
        m[0x10] = (m[0x08] - m[0x0F]) & 0xFF
        m[0x11] = (m[0x10] ^ m[0x07]) & 0xFF
        m[0x12] = (m[0x0F] ^ 0xFF) & 0xFF
        m[0x13] = (m[0x03] ^ 0x10) & 0xFF
        m[0x14] = (m[0x04] - 0x32) & 0xFF
        m[0x15] = (m[0x05] + 0xED) & 0xFF
        m[0x16] = (m[0x06] ^ 0xF3) & 0xFF
        m[0x17] = (m[0x13] - m[0x0F]) & 0xFF
        m[0x18] = (m[0x15] + m[0x07]) & 0xFF
        m[0x19] = (0x21 - m[0x13]) & 0xFF
        m[0x1A] = (m[0x14] ^ m[0x17]) & 0xFF
        m[0x1B] = (m[0x16] + m[0x16]) & 0xFF
        m[0x1C] = (m[0x17] + 0x44) & 0xFF
        m[0x1D] = (m[0x03] + m[0x04]) & 0xFF
        m[0x1E] = (m[0x05] - m[0x16]) & 0xFF
        m[0x1F] = (m[0x1D] ^ m[0x13]) & 0xFF

        # Generate video mask 2 and audio mask
        table = b"URUC"
        for i in range(MASK_SIZE):
            self.video_mask2[i] = (m[i] ^ 0xFF) & 0xFF
            self.audio_mask[i] = table[(i >> 1) & 3] if (i & 1) else (m[i] ^ 0xFF)

    def _decrypt_video(self, data: bytearray) -> None:
        """Decrypt video chunk in-place."""
        size = len(data) - VIDEO_OFFSET
        if size < MIN_VIDEO_SIZE:
            return

        mask = bytearray(self.video_mask2)

        # Decrypt from offset 0x100 onwards
        for i in range(0x100, size):
            idx = i & 0x1F
            pos = i + VIDEO_OFFSET
            encrypted_byte = data[pos]
            data[pos] ^= mask[idx]
            mask[idx] = encrypted_byte ^ self.video_mask2[idx]

        # Switch to mask1 and decrypt first 0x100 bytes
        mask[:MASK_SIZE] = self.video_mask1[:MASK_SIZE]
        for i in range(0x100):
            idx = i & 0x1F
            pos = i + VIDEO_OFFSET
            pos2 = 0x100 + i + VIDEO_OFFSET
            mask[idx] ^= data[pos2]
            data[pos] ^= mask[idx]

    def _decrypt_audio(self, data: bytearray) -> None:
        """Decrypt audio chunk in-place."""
        size = len(data) - AUDIO_OFFSET
        for i in range(size):
            data[i + AUDIO_OFFSET] ^= self.audio_mask[i & 0x1F]

    def _open_stream(self, file_path: Path, streams: dict, paths: dict, stream_type: str) -> BinaryIO:
        """Open or retrieve existing file stream."""
        path_str = str(file_path)
        if path_str not in streams:
            streams[path_str] = open(file_path, "wb")
            paths.setdefault(stream_type, []).append(path_str)
        return streams[path_str]

    def demux(self, output_path: Path) -> dict[str, list[str]]:
        """Demux USM file and extract streams."""
        base_name = self.file_path.stem
        audio_streams = {}
        file_paths = {}

        # Video parameters
        ivf_writer = None
        video_width = None
        video_height = None
        video_framerate = 30000
        video_timescale = 1000
        frame_timestamp = 0

        with open(self.file_path, "rb") as fp:
            while True:
                # Read chunk header
                header_data = fp.read(HEADER_SIZE)
                if len(header_data) < HEADER_SIZE:
                    break

                header = ChunkHeader.from_bytes(header_data)

                # Read chunk data
                data_size = header.data_size - header.data_offset - header.padding_size
                fp.seek(header.data_offset - 0x18, 1)
                data = bytearray(fp.read(data_size))
                fp.seek(header.padding_size, 1)

                # Process chunk based on signature
                if header.signature == SIG_CRID:
                    pass  # Container ID chunk, skip
                elif header.signature == SIG_VIDEO:
                    if header.data_type == 0:
                        self._decrypt_video(data)

                        # Extract frame data (skip VIDEO_OFFSET header bytes)
                        frame_data = bytes(data[VIDEO_OFFSET:])

                        # Parse dimensions from first keyframe
                        if video_width is None or video_height is None:
                            dims = parse_vp9_dimensions(frame_data)
                            if dims:
                                video_width, video_height = dims
                                # Use frame_rate from header if available
                                if header.frame_rate > 0:
                                    video_framerate = header.frame_rate
                                    video_timescale = 1000

                                # Initialize IVF writer now that we have dimensions
                                file_path = output_path.joinpath(f"{base_name}.ivf")
                                ivf_writer = IVFWriter(
                                    file_path,
                                    video_width,
                                    video_height,
                                    video_framerate,
                                    video_timescale
                                )
                                ivf_writer.__enter__()
                                file_paths.setdefault("ivf", []).append(str(file_path))
                                typer.echo(f"Video dimensions: {video_width}x{video_height}, FPS: {video_framerate}/{video_timescale}")

                        # Write frame with IVF header
                        if ivf_writer:
                            ivf_writer.write_frame(frame_data, frame_timestamp)
                            frame_timestamp += 1

                elif header.signature == SIG_AUDIO:
                    if header.data_type == 0:
                        file_path = output_path.joinpath(f"{base_name}_{header.channel_no}.hca")
                        stream = self._open_stream(file_path, audio_streams, file_paths, "hca")
                        stream.write(data)
                elif header.signature == SIG_CUE:
                    pass  # Cue point chunk, not needed
                else:
                    typer.echo(f"Unknown signature {header.signature}")

        # Close IVF writer (updates frame count in header)
        if ivf_writer:
            ivf_writer.__exit__(None, None, None)

        # Close audio streams
        for stream in audio_streams.values():
            stream.close()

        return file_paths
