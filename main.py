"""Charlotte - USM video file demuxer and converter.

A tool for extracting and converting video/audio from CRI Middleware USM files.
"""

from pathlib import Path
from typing import Annotated

import typer

from decoders.ass import ASS
from decoders.hca import HCA
from decoders.ivf import IVF
from decoders.usm import USM
from utils.keys import get_decryption_key


app = typer.Typer(help="USM video file demuxer and converter")


def collect_files(input_path: str, extension: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]

    if path.is_dir():
        files = list(path.glob(f"*.{extension}"))
        if not files:
            typer.echo(f"No .{extension} files found in directory", err=True)
            raise typer.Exit(1)
        return files

    typer.echo(f"Error: {input_path} is not a valid file or directory", err=True)
    raise typer.Exit(1)


@app.command()
def demux(
    input_path: Annotated[
        str, typer.Argument(help="USM file or directory containing USM files")
    ],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output directory")
    ] = ".",
) -> None:
    """Demux USM file(s) and extract video/audio tracks."""
    usm_files = collect_files(input_path, "usm")
    typer.echo(f"Found {len(usm_files)} USM file(s).")

    for usm_file in usm_files:
        typer.echo(f"\nProcessing: {usm_file.name}")
        key1, key2 = get_decryption_key(usm_file.name)
        usm = USM(usm_file, key1, key2)
        file_paths = usm.demux(output_dir=output)

        # Display extracted files
        if "ivf" in file_paths:
            typer.echo(f"  Video: {', '.join(file_paths['ivf'])}")
        if "hca" in file_paths:
            typer.echo(f"  Audio: {', '.join(file_paths['hca'])}")


@app.command()
def convert_hca(
    input_path: Annotated[
        str, typer.Argument(help="HCA file or directory containing HCA files")
    ],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output directory")
    ] = ".",
    key1: Annotated[
        int | None, typer.Option("--key1", help="Decryption key1 (uint64)")
    ] = None,
    key2: Annotated[
        int | None, typer.Option("--key2", help="Decryption key2 (uint16)")
    ] = None,
) -> None:
    """Convert HCA audio file(s) to WAV format."""
    hca_files = collect_files(input_path, "hca")
    typer.echo(f"Found {len(hca_files)} HCA file(s)")

    for hca_file in hca_files:
        typer.echo(f"\nConverting: {hca_file.name}")
        hca = HCA(str(hca_file), key1, key2)
        hca.decrypt()
        # Use fast ffmpeg decoder (default)
        output_wav = hca.convert_to_wav_ffmpeg(output_dir=output)

        typer.echo(f"  Output: {output_wav}")


@app.command()
def convert_ivf(
    input_path: Annotated[
        str, typer.Argument(help="IVF file or directory containing IVF files")
    ],
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Output file or directory")
    ] = None,
    codec: Annotated[
        str, typer.Option("--codec", "-c", help="Video codec")
    ] = "libx265",
    crf: Annotated[
        int, typer.Option("--crf", help="Constant Rate Factor (0-51, lower=better)")
    ] = 25,
    preset: Annotated[
        str, typer.Option("--preset", "-p", help="Encoding preset")
    ] = "slower",
    pix_fmt: Annotated[
        str, typer.Option("--pix-fmt", help="Pixel format")
    ] = "yuv420p10le",
) -> None:
    """Convert IVF video file(s) to MP4 format with optional re-encoding."""
    ivf_files = collect_files(input_path, "ivf")
    typer.echo(f"Found {len(ivf_files)} IVF file(s)")

    for ivf_file in ivf_files:
        typer.echo(f"\nConverting: {ivf_file.name}")
        ivf = IVF(str(ivf_file))

        # Determine output path
        if output and len(ivf_files) > 1:
            # Multiple files: treat output as directory
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
            output_mp4 = str(output_dir / ivf_file.with_suffix(".mp4").name)
        elif output:
            # Single file: treat output as file path
            output_mp4 = output
        else:
            # No output specified: use input filename with .mp4 extension
            output_mp4 = None

        try:
            output_path = ivf.convert_to_mp4(
                output_path=output_mp4,
                codec=codec,
                crf=crf,
                preset=preset,
                pixel_format=pix_fmt,
            )
            typer.echo(f"  Output: {output_path}")
        except Exception as e:
            typer.echo(f"  Error: {e}", err=True)


@app.command()
def convert_srt(
    input_path: Annotated[
        str, typer.Argument(help="SRT/TXT file or directory containing subtitle files")
    ],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output directory")
    ] = ".",
    lang: Annotated[
        str, typer.Option("--lang", "-l", help="Language code (JP, EN, etc.)")
    ] = "EN",
    style: Annotated[
        str | None, typer.Option("--style", help="Custom ASS style line")
    ] = None,
) -> None:
    """Convert SRT/TXT subtitle file(s) to ASS format."""
    # Collect files with multiple extensions
    path = Path(input_path)
    subtitle_files = []

    if path.is_file():
        subtitle_files = [path]
    elif path.is_dir():
        for ext in ("srt", "txt"):
            subtitle_files.extend(path.glob(f"*.{ext}"))
        if not subtitle_files:
            typer.echo("No .srt or .txt files found in directory", err=True)
            raise typer.Exit(1)
    else:
        typer.echo(f"Error: {input_path} is not a valid file or directory", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(subtitle_files)} subtitle file(s)")

    for sub_file in subtitle_files:
        typer.echo(f"\nConverting: {sub_file.name}")
        try:
            ass = ASS(str(sub_file), lang, style)

            # Skip if already in ASS format
            if ass.is_ass():
                typer.echo(f"  Already in ASS format, skipping")
                continue

            if ass.parse_srt():
                output_file = ass.convert_to_ass(output_dir=output)
                typer.echo(f"  Output: {output_file}")
            else:
                typer.echo(f"  Failed to parse subtitle file", err=True)
        except Exception as e:
            typer.echo(f"  Error: {e}", err=True)

@app.command()
def mux(
    input_path: Annotated[
        str, typer.Argument(help="Directory containing extracted video/audio/subtitle files")
    ],
    basename: Annotated[
        str, typer.Argument(help="Base filename (without extension)")
    ],
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Output MKV file path")
    ] = None,
    audio_langs: Annotated[
        str, typer.Option("--audio-langs", "-al", help="Comma-separated audio language codes (e.g., 'en,ja')")
    ] = "en,ja,ko,zh",
    subtitle_dir: Annotated[
        str | None, typer.Option("--subtitle-dir", help="Directory containing subtitle folders (e.g., EN/, JP/, etc.)")
    ] = None,
    font_ja: Annotated[
        str | None, typer.Option("--font-ja", help="Path to Japanese font file (ja-jp.ttf)")
    ] = None,
    font_zh: Annotated[
        str | None, typer.Option("--font-zh", help="Path to Chinese font file (zh-cn.ttf)")
    ] = None,
) -> None:
    """Mux video, audio, and subtitle files into a single MKV container using ffmpeg."""
    from utils.mux import FFmpegMuxer

    input_dir = Path(input_path)
    if not input_dir.is_dir():
        typer.echo(f"Error: {input_path} is not a valid directory", err=True)
        raise typer.Exit(1)

    # Determine output path
    if output is None:
        output = str(input_dir / f"{basename}.mkv")

    typer.echo(f"Muxing files for: {basename}")

    try:
        muxer = FFmpegMuxer(output)

        # Add video track
        video_file = input_dir / f"{basename}.ivf"
        if video_file.exists():
            muxer.add_video_track(str(video_file))
            typer.echo(f"  Video: {video_file.name}")
        else:
            typer.echo(f"Error: Video file not found: {video_file}", err=True)
            raise typer.Exit(1)

        # Add audio tracks
        audio_lang_list = [lang.strip() for lang in audio_langs.split(",")]
        audio_files = sorted(input_dir.glob(f"{basename}_*.wav"))

        for audio_file in audio_files:
            # Extract language number from filename (e.g., basename_0.wav -> 0)
            try:
                lang_num = int(audio_file.stem.split("_")[-1])
                if lang_num < len(FFmpegMuxer.AUDIO_LANG):
                    lang_code = FFmpegMuxer.AUDIO_LANG[lang_num][1]
                    if lang_code in audio_lang_list:
                        muxer.add_audio_track(str(audio_file), lang_num)
                        typer.echo(f"  Audio: {audio_file.name} ({FFmpegMuxer.AUDIO_LANG[lang_num][0]})")
            except (ValueError, IndexError):
                typer.echo(f"  Warning: Skipping audio file with invalid format: {audio_file.name}")

        # Add subtitle tracks
        if subtitle_dir:
            subtitle_path = Path(subtitle_dir)
            if not subtitle_path.exists():
                typer.echo(f"Warning: Subtitle directory not found: {subtitle_dir}", err=True)
            else:
                # Search for subtitle files in language subdirectories
                subtitle_count = 0
                for lang_code, (iso_code, lang_name) in FFmpegMuxer.SUBS_LANG.items():
                    # Look for subtitle files in language subdirectory
                    lang_dir = subtitle_path / lang_code
                    if not lang_dir.exists():
                        continue

                    # Search for subtitle files matching basename
                    srt_files = list(lang_dir.glob(f"{basename}_{lang_code}.*"))
                    if not srt_files:
                        continue

                    # Use the first matching file (prioritize .ass, then .srt)
                    sub_file = None
                    for f in srt_files:
                        if f.suffix.lower() in ['.ass', '.srt', '.txt']:
                            sub_file = f
                            if f.suffix.lower() == '.ass':
                                break  # Prefer .ass files

                    if not sub_file:
                        continue

                    # Convert to ASS if needed
                    if sub_file.suffix.lower() != '.ass':
                        typer.echo(f"  Converting subtitle: {sub_file.name}")
                        ass = ASS(str(sub_file), lang_code)
                        if ass.parse_srt():
                            ass_file = ass.convert_to_ass(output_dir=str(input_dir))
                            sub_file = Path(ass_file)
                        else:
                            typer.echo(f"  Warning: Failed to parse {sub_file.name}", err=True)
                            continue

                    # Add subtitle track
                    muxer.add_subtitles_track(str(sub_file), lang_code)
                    typer.echo(f"  Subtitle: {sub_file.name} ({lang_name})")
                    subtitle_count += 1

                if subtitle_count == 0:
                    typer.echo(f"  Warning: No subtitles found for {basename}")
                else:
                    typer.echo(f"  Added {subtitle_count} subtitle track(s)")

        # Add font attachments
        if font_ja and Path(font_ja).exists():
            muxer.add_attachment(font_ja, "Japanese Font")
            typer.echo(f"  Attachment: {Path(font_ja).name}")
        elif Path("ja-jp.ttf").exists():
            muxer.add_attachment("ja-jp.ttf", "Japanese Font")
            typer.echo(f"  Attachment: ja-jp.ttf")

        if font_zh and Path(font_zh).exists():
            muxer.add_attachment(font_zh, "Chinese Font")
            typer.echo(f"  Attachment: {Path(font_zh).name}")
        elif Path("zh-cn.ttf").exists():
            muxer.add_attachment("zh-cn.ttf", "Chinese Font")
            typer.echo(f"  Attachment: zh-cn.ttf")

        # Perform muxing
        muxer.merge()
        typer.echo(f"\nMuxing complete: {output}")

    except Exception as e:
        typer.echo(f"Error during muxing: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
