"""FFmpeg-based muxer for combining video, audio, and subtitle streams into MKV container.

This module provides functionality to mux VP9 video (IVF), WAV audio, and ASS subtitles
into a single Matroska (MKV) container file using ffmpeg.
"""

import subprocess
from pathlib import Path


class FFmpegMuxer:
    """FFmpeg-based muxer for creating MKV containers.

    Combines video, audio, subtitle, and attachment files into a single MKV file
    using ffmpeg with proper metadata and track mapping.

    Args:
        output_path: Path to the output MKV file
        ffmpeg_path: Path to ffmpeg executable (default: searches in PATH and project root)
    """

    # Language mappings (index -> (name, ISO 639-2 code))
    AUDIO_LANG = [
        ("English", "en"),
        ("Japanese", "ja"),
        ("Korean", "ko"),
        ("Chinese", "zh"),
    ]

    # Subtitle language mappings (Genshin Impact code -> (ISO 639-2, name))
    SUBS_LANG = {
        "EN": ("en", "English"),
        "CHS": ("zh", "Chinese (Simplified)"),
        "CHT": ("zh", "Chinese (Traditional)"),
        "DE": ("de", "German"),
        "ES": ("es", "Spanish"),
        "FR": ("fr", "French"),
        "ID": ("id", "Indonesian"),
        "IT": ("it", "Italian"),
        "JP": ("ja", "Japanese"),
        "KR": ("ko", "Korean"),
        "PT": ("pt", "Portuguese"),
        "RU": ("ru", "Russian"),
        "TH": ("th", "Thai"),
        "TR": ("tr", "Turkish"),
        "VI": ("vi", "Vietnamese"),
    }

    def __init__(self, output_path: str, ffmpeg_path: str | None = None):
        self.output_path = output_path
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()

        # Track counters
        self.video_count = 0
        self.audio_count = 0
        self.subs_count = 0
        self.attachment_count = 0

        # Command components
        self.input_options: list[str] = []
        self.map_options: list[str] = []
        self.metadata_options: list[str] = []

    @staticmethod
    def _find_ffmpeg() -> str:
        """Find ffmpeg executable in project root or PATH."""
        project_ffmpeg = Path("ffmpeg.exe")
        if project_ffmpeg.exists():
            return str(project_ffmpeg.absolute())
        return "ffmpeg"

    def add_video_track(self, video_file: str) -> None:
        """Add video track to the mux.

        Args:
            video_file: Path to IVF video file

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        if not Path(video_file).exists():
            raise FileNotFoundError(f"Video file {video_file} not found.")

        name = Path(video_file).stem
        self.input_options.append(f"-i")
        self.input_options.append(video_file)

        track_index = self.video_count + self.audio_count + self.subs_count
        self.map_options.extend(["-map", str(track_index)])
        self.metadata_options.extend(
            [
                f"-metadata:s:v:{self.video_count}",
                "language=und",
                f"-metadata:s:v:{self.video_count}",
                f"title={name}",
            ]
        )
        self.video_count += 1

    def add_audio_track(self, audio_file: str, lang: int) -> None:
        """Add audio track to the mux.

        Args:
            audio_file: Path to WAV audio file
            lang: Language index (0-3) corresponding to AUDIO_LANG mapping

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If language index is invalid
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file {audio_file} not found.")
        if not 0 <= lang < len(self.AUDIO_LANG):
            raise ValueError(f"Language number {lang} not supported")

        lang_name, lang_code = self.AUDIO_LANG[lang]

        self.input_options.append("-i")
        self.input_options.append(audio_file)

        track_index = self.video_count + self.audio_count + self.subs_count
        self.map_options.extend(["-map", str(track_index)])
        self.metadata_options.extend(
            [
                f"-metadata:s:a:{self.audio_count}",
                f"language={lang_code}",
                f"-metadata:s:a:{self.audio_count}",
                f"title={lang_name}",
            ]
        )
        self.audio_count += 1

    def add_subtitles_track(self, sub_file: str, language: str) -> None:
        """Add subtitle track to the mux.

        Args:
            sub_file: Path to ASS subtitle file
            language: Language code (e.g., "EN", "JP", "CHS")

        Raises:
            FileNotFoundError: If subtitle file doesn't exist
            ValueError: If language code is not supported
        """
        if language not in self.SUBS_LANG:
            raise ValueError(f"Language code {language} isn't supported...")

        self.input_options.append("-i")
        self.input_options.append(sub_file)

        lang_code, lang_name = self.SUBS_LANG[language]

        track_index = self.video_count + self.audio_count + self.subs_count
        self.map_options.extend(["-map", str(track_index)])
        self.metadata_options.extend(
            [
                f"-metadata:s:s:{self.subs_count}",
                f"language={lang_code}",
                f"-metadata:s:s:{self.subs_count}",
                f"title={lang_name}",
            ]
        )
        self.subs_count += 1

    def add_attachment(self, attachment_file: str, description: str) -> None:
        """Add attachment (e.g., font file) to the mux.

        Args:
            attachment_file: Path to attachment file
            description: Description of the attachment

        Raises:
            FileNotFoundError: If attachment file doesn't exist
        """
        if not Path(attachment_file).exists():
            raise FileNotFoundError(f"Attachment file {attachment_file} not found.")

        self.input_options.append("-attach")
        self.input_options.append(attachment_file)
        self.metadata_options.extend(
            [
                f"-metadata:s:t:{self.attachment_count}",
                "mimetype=application/x-truetype-font",
                f"-metadata:s:t:{self.attachment_count}",
                f"description={description}",
            ]
        )
        self.attachment_count += 1

    def merge(self) -> None:
        """Execute the muxing process.

        Builds and runs the ffmpeg command to create the final MKV file.

        Raises:
            subprocess.CalledProcessError: If ffmpeg command fails
            FileNotFoundError: If ffmpeg executable is not found
        """
        # Build command
        cmd = [self.ffmpeg_path, "-y", "-loglevel", "error", "-nostats"]
        cmd.extend(self.input_options)
        cmd.extend(self.map_options)
        cmd.extend(self.metadata_options)
        cmd.extend(["-c", "copy", self.output_path])

        print(f"Running ffmpeg muxing...")

        # Execute command
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            if result.stderr:
                print(f"ffmpeg output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error during muxing: {e}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            raise FileNotFoundError(
                "ffmpeg not found. Please install ffmpeg to mux videos."
            )
