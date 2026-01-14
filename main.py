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


def collect_files(input_path: str | Path, extension: str) -> list[Path]:
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = list(input_path.glob(f"*.{extension}"))
        if not files:
            typer.echo(f"No .{extension} files found in directory", err=True)
            raise typer.Exit(1)
        return files

    typer.echo(f"Error: {input_path} is not a valid file or directory", err=True)
    raise typer.Exit(1)

@app.command()
def demux(
    input_path: Annotated[
        str, typer.Argument(help="USM file or directory containing USM files.")
    ],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output directory.")
    ] = "output",
    no_cleanup: Annotated[
        bool, typer.Option("--no-cleanup", "-nc", help="Do not delete decoded .ivf and .hca files when done.")
    ] = False,
) -> None:
    """Demux USM file(s) and extract video/audio tracks."""
    usm_files = collect_files(input_path, "usm")
    typer.echo(f"Found {len(usm_files)} USM file(s).")
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)

    for usm_file in usm_files:
        typer.echo(f"\nProcessing: {usm_file.name}")
        key1, key2 = get_decryption_key(usm_file.name)
        usm = USM(usm_file, key1, key2)
        video_path = output_path.joinpath(usm_file.stem)
        video_path.mkdir(exist_ok=True)
        file_paths = usm.demux(output_path=video_path)

        # Display extracted files
        if "ivf" in file_paths:
            typer.echo(f"Extracted IVF: {', '.join(file_paths['ivf'])}")
        if "hca" in file_paths:
            typer.echo(f"Extracted HCA: {', '.join(file_paths['hca'])}")

        process_hca(video_path, key1, key2)
        process_ivf(video_path)
        mux(video_path)

def process_hca(output_path: Path, key1: int, key2: int) -> None:
    """Decrypt HCA files and convert to FLAC."""
    hca_files = collect_files(output_path, "hca")
    for hca_file in hca_files:
        hca = HCA(str(hca_file), key1, key2)
        hca.decrypt()
        flac_file = hca.convert_to_flac(output_path=output_path)
        typer.echo(f"Converted FLAC: {flac_file}")

def process_ivf(output_path: Path) -> None:
    """Convert IVF files to MP4."""
    ivf_files = collect_files(output_path, "ivf")
    for ivf_file in ivf_files:
        ivf = IVF(str(ivf_file))
        mp4_file = ivf.convert_to_mp4(output_path=output_path)
        typer.echo(f"Converted IVF: {mp4_file}")

# def process_srt() -> None:
#     """Convert SRT/TXT subtitle file(s) to ASS format."""
#     # Collect files with multiple extensions
#     path = Path.cwd()
#     subtitle_files = []
#
#     if path.is_file():
#         subtitle_files = [path]
#     elif path.is_dir():
#         for ext in ("srt", "txt"):
#             subtitle_files.extend(path.glob(f"*.{ext}"))
#         if not subtitle_files:
#             typer.echo("No .srt or .txt files found in directory", err=True)
#             raise typer.Exit(1)
#     else:
#         typer.echo(f"Error: {input_path} is not a valid file or directory", err=True)
#         raise typer.Exit(1)
#
#     typer.echo(f"Found {len(subtitle_files)} subtitle file(s)")
#
#     for sub_file in subtitle_files:
#         typer.echo(f"\nConverting: {sub_file.name}")
#         try:
#             ass = ASS(str(sub_file), lang, style)
#
#             # Skip if already in ASS format
#             if ass.is_ass():
#                 typer.echo(f"  Already in ASS format, skipping")
#                 continue
#
#             if ass.parse_srt():
#                 output_file = ass.convert_to_ass(output_dir=output)
#                 typer.echo(f"  Output: {output_file}")
#             else:
#                 typer.echo(f"  Failed to parse subtitle file", err=True)
#         except Exception as e:
#             typer.echo(f"  Error: {e}", err=True)

def mux(output_path: Path) -> None:
    """Mux MP4, FLAC, and fonts into MKV container using mkvmerge."""
    import subprocess

    # Collect video and audio files
    mp4_files = list(output_path.glob("*.mp4"))
    flac_files = list(output_path.glob("*.flac"))

    if not mp4_files:
        typer.echo("No MP4 files found to mux", err=True)
        return

    # Get font files from font folder
    font_folder = output_path / "font"
    font_files = []
    if font_folder.exists() and font_folder.is_dir():
        for ext in ("ttf", "otf", "ttc"):
            font_files.extend(font_folder.glob(f"*.{ext}"))

    # Mux each MP4 with corresponding FLAC and fonts
    for mp4_file in mp4_files:
        output_mkv = output_path / f"{mp4_file.stem}.mkv"

        # Build mkvmerge command
        cmd = ["mkvmerge", "-o", str(output_mkv), str(mp4_file)]

        # Add audio tracks
        for flac_file in flac_files:
            cmd.extend([str(flac_file)])

        # Add font attachments
        for font_file in font_files:
            cmd.extend(["--attachment-mime-type", "application/x-truetype-font",
                       "--attach-file", str(font_file)])

        typer.echo(f"Muxing: {output_mkv.name}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            typer.echo(f"Created: {output_mkv}")

            # Cleanup intermediate files if requested
            # if not no_cleanup:
            #     mp4_file.unlink()
            #     for flac_file in flac_files:
            #         flac_file.unlink()
            #     typer.echo("Cleaned up intermediate files")
        except subprocess.CalledProcessError as e:
            typer.echo(f"Error muxing {mp4_file.name}: {e.stderr}", err=True)
        except FileNotFoundError:
            typer.echo("Error: mkvmerge not found. Please install mkvtoolnix.", err=True)
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
