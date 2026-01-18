"""Charlotte - USM video file demuxer and converter.

A tool for extracting and converting video/audio from CRI Middleware USM files.
"""
import re
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

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
        process_srt()
        mux(video_path)

def process_hca(output_path: Path, key1: int, key2: int) -> None:
    """Decrypt HCA files and convert to FLAC."""
    hca_files = collect_files(output_path, "hca")
    for hca_file in hca_files:
        hca = HCA(str(hca_file), key1, key2)
        hca.decrypt()
        flac_file = hca.convert_to_flac(output_path=output_path)
        typer.echo(f"Converted FLAC: {flac_file}")

def process_srt(output_path: Path) -> None:
    """Convert SRT/TXT subtitle file(s) to ASS format."""
    # Collect files with multiple extensions
    path = Path.cwd()
    subtitle_files = []

    if path.is_file():
        subtitle_files = [path]
    elif path.is_dir():
        subtitle_files.extend(path.glob(f"*.srt"))
        if not subtitle_files:
            typer.echo("No .srt files found in directory.", err=True)
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

def mux(output_path: Path) -> None:
    """Mux IVF video and FLAC audio into MKV container using mkvmerge."""
    # Collect video and audio files
    ivf_file = output_path.joinpath(output_path.stem + ".ivf")
    flac_files = list(output_path.glob("*.flac"))

    if not ivf_file.exists():
        typer.echo(f"IVF file not found: {ivf_file}", err=True)
        return

    if not flac_files:
        typer.echo("No FLAC files found to mux.", err=True)
        return

    output_mkv = output_path / f"{output_path.stem}.mkv"

    # Build mkvmerge command
    cmd = ["mkvmerge", "-o", str(output_mkv), str(ivf_file)]

    # Add audio tracks with language tags
    langs = {
        0: "zh",
        1: "en",
        2: "ja",
        3: "ko",
    }
    flac_files.sort(key=lambda x: 0 if "_2.flac" in str(x) else 1)
    for i, flac_file in enumerate(flac_files):
        cmd.extend([
            "--language", f"0:{langs[i]}",
            "--default-track-flag", f"0:{1 if langs[i] == "ja" else 0}",
            str(flac_file),
        ])

    typer.echo(" ".join(cmd))
    typer.echo(f"Muxing: {output_mkv.name}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        stdout, stderr = process.communicate()
        return_code = process.returncode

        if return_code != 0:
            typer.echo(f"Error muxing video: mkvmerge exited with code {return_code}")
            if stdout:
                typer.echo(f"stdout: {stdout}")
            if stderr:
                typer.echo(f"stderr: {stderr}")
            raise typer.Exit(1)

        typer.echo(f"Created: {output_mkv}")

        # Cleanup intermediate files if requested
        # if not no_cleanup:
        #     mkv_file.unlink()
        #     for flac_file in flac_files:
        #         flac_file.unlink()
        #     typer.echo("Cleaned up intermediate files")
    except FileNotFoundError:
        typer.echo(
            "mkvmerge not found. Place mkvmerge in the root directory and try again."
        )
        raise typer.Exit(1) from None

if __name__ == "__main__":
    app()
