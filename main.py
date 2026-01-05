from pathlib import Path
from typing import Annotated, Optional

import typer

from decoders.hca import HCA
from decoders.usm import USM

app = typer.Typer()


@app.command()
def demux(
    input_path: Annotated[str, typer.Argument(help="USM file or directory containing USM files")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = ".",
    key1: Annotated[Optional[str], typer.Option("--key1", "-b", help="4 lower bytes of decryption key (hex)")] = None,
    key2: Annotated[Optional[str], typer.Option("--key2", "-a", help="4 higher bytes of decryption key (hex)")] = None,
    video: Annotated[bool, typer.Option("--video/--no-video", help="Extract video")] = True,
    audio: Annotated[bool, typer.Option("--audio/--no-audio", help="Extract audio")] = True,
) -> None:
    """Demux USM file(s) and extract video/audio tracks"""
    input_p = Path(input_path)

    # Default keys (can be overridden)
    k1 = bytes.fromhex(key1) if key1 else bytes.fromhex("00000000")
    k2 = bytes.fromhex(key2) if key2 else bytes.fromhex("00000000")

    # Collect USM files
    usm_files = []
    if input_p.is_file():
        usm_files.append(input_p)
    elif input_p.is_dir():
        usm_files = list(input_p.glob("*.usm"))
    else:
        typer.echo(f"Error: {input_path} is not a valid file or directory", err=True)
        raise typer.Exit(1)

    if not usm_files:
        typer.echo("No USM files found", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(usm_files)} USM file(s)")

    for usm_file in usm_files:
        typer.echo(f"\nProcessing: {usm_file.name}")
        usm = USM(str(usm_file), k1, k2)
        file_paths = usm.demux(video_extract=video, audio_extract=audio, output_dir=output)

        # Print extracted files
        if "ivf" in file_paths:
            typer.echo(f"  Video: {', '.join(file_paths['ivf'])}")
        if "hca" in file_paths:
            typer.echo(f"  Audio: {', '.join(file_paths['hca'])}")


@app.command()
def convert_hca(
    input_path: Annotated[str, typer.Argument(help="HCA file or directory containing HCA files")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = ".",
    key1: Annotated[Optional[int], typer.Option("--key1", help="Decryption key1 (uint64)")] = None,
    key2: Annotated[Optional[int], typer.Option("--key2", help="Decryption key2 (uint16)")] = None,
) -> None:
    """Convert HCA file(s) to WAV format"""
    input_p = Path(input_path)

    # Collect HCA files
    hca_files = []
    if input_p.is_file():
        hca_files.append(input_p)
    elif input_p.is_dir():
        hca_files = list(input_p.glob("*.hca"))
    else:
        typer.echo(f"Error: {input_path} is not a valid file or directory", err=True)
        raise typer.Exit(1)

    if not hca_files:
        typer.echo("No HCA files found", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(hca_files)} HCA file(s)")

    for hca_file in hca_files:
        typer.echo(f"\nConverting: {hca_file.name}")
        hca = HCA(str(hca_file), key1, key2)
        hca.decrypt()
        output_wav = hca.convert_to_wav(output_dir=output)
        typer.echo(f"  Output: {output_wav}")


if __name__ == "__main__":
    app()
