"""USM (CRI Sofdec) container format demuxer.

This module demuxes CRI Middleware's USM container files, extracting:
- Video streams (VP9 codec in IVF format)
- Audio streams (HCA format)

USM files are commonly used in video games for cutscenes and movies.
"""
import socket
import struct
from pathlib import Path


class Header:
    """USM chunk header."""
    def __init__(self):
        self.signature: int = 0
        self.data_size: int = 0
        self.data_offset: int = 0
        self.padding_size: int = 0
        self.channel_no: int = 0
        self.data_type: int = 0
        self.frame_time: int = 0
        self.frame_rate: int = 0


class USM:
    """USM container demuxer.

    Demuxes USM files and extracts video (VP9/IVF) and audio (HCA) streams.
    Handles encrypted USM files using provided keys.

    Args:
        file_path: Path to the USM file
        key1: Decryption key (4 bytes) for video/audio
        key2: Decryption key (4 bytes) for video/audio
    """

    def __init__(self, file_path: str, key1: bytes, key2: bytes):
        self.file_path = Path(file_path)
        self.filename = self.file_path.name
        self.key1 = key1
        self.key2 = key2
        self.video_mask1 = bytearray(0x20)
        self.video_mask2 = bytearray(0x20)
        self.audio_mask = bytearray(0x20)
        self._init_mask(key1, key2)

    def _init_mask(self, key1: bytes, key2: bytes):
        self.video_mask1[0x00] = key1[0]
        self.video_mask1[0x01] = key1[1]
        self.video_mask1[0x02] = key1[2]
        self.video_mask1[0x03] = (key1[3] - 0x34) & 0xFF
        self.video_mask1[0x04] = (key2[0] + 0xF9) & 0xFF
        self.video_mask1[0x05] = (key2[1] ^ 0x13) & 0xFF
        self.video_mask1[0x06] = (key2[2] + 0x61) & 0xFF
        self.video_mask1[0x07] = (self.video_mask1[0x00] ^ 0xFF) & 0xFF
        self.video_mask1[0x08] = (self.video_mask1[0x02] + self.video_mask1[0x01]) & 0xFF
        self.video_mask1[0x09] = (self.video_mask1[0x01] - self.video_mask1[0x07]) & 0xFF
        self.video_mask1[0x0A] = (self.video_mask1[0x02] ^ 0xFF) & 0xFF
        self.video_mask1[0x0B] = (self.video_mask1[0x01] ^ 0xFF) & 0xFF
        self.video_mask1[0x0C] = (self.video_mask1[0x0B] + self.video_mask1[0x09]) & 0xFF
        self.video_mask1[0x0D] = (self.video_mask1[0x08] - self.video_mask1[0x03]) & 0xFF
        self.video_mask1[0x0E] = (self.video_mask1[0x0D] ^ 0xFF) & 0xFF
        self.video_mask1[0x0F] = (self.video_mask1[0x0A] - self.video_mask1[0x0B]) & 0xFF
        self.video_mask1[0x10] = (self.video_mask1[0x08] - self.video_mask1[0x0F]) & 0xFF
        self.video_mask1[0x11] = (self.video_mask1[0x10] ^ self.video_mask1[0x07]) & 0xFF
        self.video_mask1[0x12] = (self.video_mask1[0x0F] ^ 0xFF) & 0xFF
        self.video_mask1[0x13] = (self.video_mask1[0x03] ^ 0x10) & 0xFF
        self.video_mask1[0x14] = (self.video_mask1[0x04] - 0x32) & 0xFF
        self.video_mask1[0x15] = (self.video_mask1[0x05] + 0xED) & 0xFF
        self.video_mask1[0x16] = (self.video_mask1[0x06] ^ 0xF3) & 0xFF
        self.video_mask1[0x17] = (self.video_mask1[0x13] - self.video_mask1[0x0F]) & 0xFF
        self.video_mask1[0x18] = (self.video_mask1[0x15] + self.video_mask1[0x07]) & 0xFF
        self.video_mask1[0x19] = (0x21 - self.video_mask1[0x13]) & 0xFF
        self.video_mask1[0x1A] = (self.video_mask1[0x14] ^ self.video_mask1[0x17]) & 0xFF
        self.video_mask1[0x1B] = (self.video_mask1[0x16] + self.video_mask1[0x16]) & 0xFF
        self.video_mask1[0x1C] = (self.video_mask1[0x17] + 0x44) & 0xFF
        self.video_mask1[0x1D] = (self.video_mask1[0x03] + self.video_mask1[0x04]) & 0xFF
        self.video_mask1[0x1E] = (self.video_mask1[0x05] - self.video_mask1[0x16]) & 0xFF
        self.video_mask1[0x1F] = (self.video_mask1[0x1D] ^ self.video_mask1[0x13]) & 0xFF

        table2 = b"URUC"
        for i in range(0x20):
            self.video_mask2[i] = (self.video_mask1[i] ^ 0xFF) & 0xFF
            if (i & 1) == 1:
                self.audio_mask[i] = table2[(i >> 1) & 3]
            else:
                self.audio_mask[i] = (self.video_mask1[i] ^ 0xFF) & 0xFF

    def _mask_video(self, data: bytearray, size: int):
        data_offset = 0x40
        size -= data_offset

        if size < 0x200:
            return

        mask = bytearray(self.video_mask2)

        for i in range(0x100, size):
            mask_idx = i & 0x1F
            data_idx = i + data_offset
            data[data_idx] ^= mask[mask_idx]
            mask[mask_idx] = (data[data_idx] ^ self.video_mask2[mask_idx]) & 0xFF

        mask[:0x20] = self.video_mask1[:0x20]

        for i in range(0x100):
            mask_idx = i & 0x1F
            data_idx = i + data_offset
            data_idx2 = 0x100 + i + data_offset

            mask[mask_idx] ^= data[data_idx2]
            data[data_idx] ^= mask[mask_idx]

    def _mask_audio(self, data: bytearray, size: int):
        data_offset = 0x140
        size -= data_offset

        for i in range(size):
            data[i + data_offset] ^= self.audio_mask[i & 0x1F]

    @staticmethod
    def _bswap(value: int, size: int = 4) -> int:
        if size == 4:
            return socket.htonl(value)
        elif size == 2:
            return socket.htons(value)
        else:
            raise ValueError(f"Unsupported size: {size}")

    def demux(
        self, video_extract: bool = True, audio_extract: bool = True, output_dir: str = "."
    ) -> dict:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        file_streams = {}
        file_paths = {}

        print(f"Demuxing {self.filename} : extracting video and audio...")

        with open(self.file_path, "rb") as fp:
            file_size = self.file_path.stat().st_size

            while file_size > 0:
                byte_block = fp.read(32)
                if len(byte_block) < 32:
                    break

                file_size -= 32

                header = Header()
                header.signature = self._bswap(struct.unpack("<I", byte_block[0:4])[0])
                header.data_size = self._bswap(struct.unpack("<I", byte_block[4:8])[0])
                header.data_offset = byte_block[9]
                header.padding_size = self._bswap(struct.unpack("<H", byte_block[10:12])[0], 2)
                header.channel_no = byte_block[12]
                header.data_type = byte_block[15]
                header.frame_time = self._bswap(struct.unpack("<I", byte_block[16:20])[0])
                header.frame_rate = self._bswap(struct.unpack("<I", byte_block[20:24])[0])

                size = header.data_size - header.data_offset - header.padding_size
                fp.seek(header.data_offset - 0x18, 1)
                data = bytearray(fp.read(size))
                fp.seek(header.padding_size, 1)
                file_size -= header.data_size - 0x18

                if header.signature == 0x43524944:
                    pass

                elif header.signature == 0x40534656:
                    if header.data_type == 0 and video_extract:
                        self._mask_video(data, size)
                        file_path = output_path / f"{self.filename[:-4]}.ivf"

                        if str(file_path) not in file_streams:
                            file_streams[str(file_path)] = open(file_path, "wb")
                            if "ivf" not in file_paths:
                                file_paths["ivf"] = []
                            file_paths["ivf"].append(str(file_path))

                        file_streams[str(file_path)].write(data)

                elif header.signature == 0x40534641:
                    if header.data_type == 0 and audio_extract:
                        file_path = output_path / f"{self.filename[:-4]}_{header.channel_no}.hca"

                        if str(file_path) not in file_streams:
                            file_streams[str(file_path)] = open(file_path, "wb")
                            if "hca" not in file_paths:
                                file_paths["hca"] = []
                            file_paths["hca"].append(str(file_path))

                        file_streams[str(file_path)].write(data)

                elif header.signature == 0x40435545:  # @CUE
                    print("@CUE field detected in USM, skipping as we don't need it")

                else:
                    print(f"Signature {header.signature} unknown, skipping...")

        for stream in file_streams.values():
            stream.close()

        return file_paths
