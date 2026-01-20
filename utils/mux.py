import subprocess

from pathlib import Path

import typer

from decoders.ass import ASS
from utils import languages
from utils.languages import AUDIO_LANGUAGES, SUBTITLES_LANGUAGES


def process_srt(file_name: str, output_path: Path) -> None:
    """Convert SRT subtitle to ASS format."""
    basename_fixes = {
        "Cs_4131904_HaiDaoChuXian_Boy": "Cs_Activity_4001103_Summertime_Boy",
        "Cs_4131904_HaiDaoChuXian_Girl": "Cs_Activity_4001103_Summertime_Girl",
        "Cs_200211_WanYeXianVideo": "Cs_DQAQ200211_WanYeXianVideo",
    }

    if file_name in basename_fixes:
        file_name = basename_fixes.get(file_name)

    subtitle_files = []
    input_path = Path.cwd().joinpath("Subtitle")
    for lang in SUBTITLES_LANGUAGES:
        lang_path = input_path.joinpath(lang)
        subtitle_path = lang_path.joinpath(f"{file_name}_{lang}.srt")
        if subtitle_path.exists():
            subtitle_files.append(subtitle_path)

    typer.echo(f"Found {len(subtitle_files)} subtitle file(s).")

    for sub_file in subtitle_files:
        lang = sub_file.stem.split("_")[-1]
        try:
            ass = ASS(str(sub_file), lang)
            if ass.parse_srt():
                ass.convert_to_ass(output_path=output_path)
            else:
                typer.echo("Failed to parse subtitle file.", err=True)
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)


def mux(output_path: Path) -> None:
    """Mux IVF video and FLAC audio into MKV container using mkvmerge."""
    # Collect video and audio files
    ivf_file = output_path.joinpath(output_path.stem + ".ivf")
    flac_files = list(output_path.glob("*.flac"))
    subtitle_files = list(output_path.joinpath("Subtitles").glob("*.ass"))

    if not ivf_file.exists():
        typer.echo(f"IVF file not found: {ivf_file}", err=True)
        return

    if not flac_files:
        typer.echo("No FLAC files found to mux.", err=True)
        return

    output_mkv = output_path / f"{output_path.stem}.mkv"

    # Build mkvmerge command
    cmd = ["mkvmerge", "-o", str(output_mkv), str(ivf_file)]

    # Put JP track with EN sub to the top.
    flac_files.sort(key=lambda x: 0 if "_2.flac" in str(x) else 1)
    subtitle_files.sort(key=lambda x: 0 if "_EN.ass" in str(x) else 1)

    for flac_file in flac_files:
        index = flac_file.stem.split("_")[-1]
        cmd.extend(
            [
                "--language",
                f"0:{AUDIO_LANGUAGES.get(index, 'und')}",
                "--default-track-flag",
                f"0:{1 if AUDIO_LANGUAGES.get(index, 'und') == 'ja' else 0}",
                str(flac_file),
            ]
        )

    # Add subtitles.
    for subtitle_file in subtitle_files:
        subtitle_lang = subtitle_file.stem.split("_")[-1]
        cmd.extend(
            [
                "--language",
                f"0:{languages.get_language(subtitle_lang)}",
                "--default-track-flag",
                f"0:{1 if '_EN' in str(subtitle_file) else 0}",
                "--forced-display-flag",
                f"0:{1 if '_EN' in str(subtitle_file) else 0}",
                str(subtitle_file),
            ]
        )

    # Attach fonts.
    font_ja = Path.cwd().joinpath("fonts").joinpath("ja-jp.ttf")
    font_zh = Path.cwd().joinpath("fonts").joinpath("zh-cn.ttf")
    cmd.extend(
        [
            "--attach-file",
            f"{font_ja}",
            "--attach-file",
            f"{font_zh}",
        ]
    )

    # typer.echo(f"Command: {' '.join(cmd)}")
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
