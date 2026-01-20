"""Charlotte - USM video file demuxer and converter.

A tool for extracting and converting video/audio from CRI Middleware USM files.
"""

from pathlib import Path
from typing import Annotated

import typer

from decoders.hca import HCA
from decoders.usm import USM
from utils.keys import get_decryption_key
from utils.mux import mux, process_srt


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
        bool,
        typer.Option(
            "--no-cleanup",
            "-nc",
            help="Do not delete decoded .ivf, .hca, and subtitle files when done.",
        ),
    ] = False,
) -> None:
    """Demux USM file(s) and extract video/audio tracks."""
    usm_files = collect_files(input_path, "usm")
    typer.echo(f"Found {len(usm_files)} USM file(s).")
    Path(output).mkdir(exist_ok=True)

    for usm_file in usm_files:
        typer.echo(f"\nProcessing: {usm_file.name}")
        key1, key2 = get_decryption_key(usm_file.name)
        usm = USM(usm_file, key1, key2)
        output_path = Path(output).joinpath(usm_file.stem)
        output_path.mkdir(exist_ok=True)
        file_paths = usm.demux(output_path=output_path)

        # Display extracted files
        if "ivf" in file_paths:
            typer.echo(f"Extracted IVF: {', '.join(file_paths['ivf'])}")
        if "hca" in file_paths:
            typer.echo(f"Extracted HCA: {', '.join(file_paths['hca'])}")

        hca_files = collect_files(output_path, "hca")
        for hca_file in hca_files:
            hca = HCA(str(hca_file), key1, key2)
            hca.decrypt()
            flac_file = hca.convert_to_flac(output_path=output_path)
            typer.echo(f"Converted FLAC: {flac_file}")

        process_srt(file_name=usm_file.stem, output_path=output_path)
        mux(output_path)

        if not no_cleanup:
            for ivf_file in collect_files(output_path, "ivf"):
                ivf_file.unlink()
            for flac_file in collect_files(output_path, "flac"):
                flac_file.unlink()
            for hca_file in collect_files(output_path, "hca"):
                hca_file.unlink()
            sub_folder = output_path.joinpath("subs")
            for subtitle_file in collect_files(sub_folder, "ass"):
                subtitle_file.unlink()
            sub_folder.rmdir()


if __name__ == "__main__":
    app()
