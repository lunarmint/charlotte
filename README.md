# Charlotte

A program for Genshin Impact that decrypts the `.usm` cutscene files into playable `.mkv` videos with official audio (EN, CN, JP, KR) and subtitles in 15 languages, with [VapourSynth](https://github.com/vapoursynth/vapoursynth) (a lossless video processing framework) suport to improve video artifacts such as banding, blocking, and chroma abberation that exists in the original cutscenes.

All cutscenes from version 1.0 to 6.3 can be decrypted.

Feel free to submit a pull request if you have missing keys.

## Why Charlotte?

Where do you think those cutscenes came from? There must be someone who recorded them for us!

This tool was heavily inspired by [GI-cutscenes](https://github.com/ToaHartor/GI-cutscenes). All the decrypt algorithm is based off of this project. GI-cutscene has not been in active development for some time now, so I wanted to rewrite it at higher level and add features down the road such as VapourSynth processing and a GUI.

## Features and Roadmap

- [x] Decrypt `.usm` files into `.ivf` video and `.hca` audio.
- [x] Subtitle support, formatting `.srt` into `.ass` format.
- [x] Font and subtitle style matching the official version.
- [x] `.hca` audio to `.flac` for archival purposes.
- [x] Mux video and audio into `.mkv`.
- [ ] Add VapourSynth processing.
- [ ] Add GUI.

## Running Charlotte

### Prerequisites
- Download `charlotte.exe` from the [releases page](https://github.com/lunarmint/charlotte/releases/latest).
- Ensure that [ffmpeg.exe](https://ffmpeg.org/download.html#build-windows) and [mkvmerge.exe](https://mkvtoolnix.download/downloads.html#windows) are present in the same directory as `charlotte.exe`.
- Clone [this repository](https://gitlab.com/Dimbreath/AnimeGameData) place the `Subtitle` directory in the same directory as `charlotte.exe`.
- To get font files, go to `[Game Directory]\Genshin Impact game\GenshinImpact_Data\StreamingAssets\MiHoYoSDKRes\HttpServerResources` and copy the `font` directory into the same directory as `charlotte.exe`.

### Usage
- Run `charlotte.exe` and select the `.usm` file or directory containing `.usm` files.

```sh
charlotte [PATH_TO_USM_FILE_OR_DIR] [OPTIONS]
```

### Example

```sh
charlotte C:\Users\Mint\Desktop\charlotte\USM\Cs_EQHDJ005_HaiDengJie_Boy.usm -nc
```

This will decrypt `Cs_EQHDJ005_HaiDengJie_Boy.usm` and output the result to `output/Cs_EQHDJ005_HaiDengJie_Boy.mkv` without cleaning up intermediate files.

### Arguments

- `PATH_TO_USM_FILE_OR_DIR`: **(Required)** Path to a single `.usm` file or a directory containing `.usm` files.

### Options

- `-o, --output [DIR]`: Output directory. Default is `output`.
- `-nc, --no-cleanup`: Do not delete intermediate decoded files (e.g., `.ivf`, `.hca`, `.ass`) after the process is complete.

## Building from Source

### Prerequisites

- **Python 3.14 or higher**
- **[uv](https://github.com/astral-sh/uv)** package manager.

### How to Build

```sh
uv run pyinstaller charlotte.spec
```

### How to Run

For external dependencies and arguments, see [Running Charlotte](#running-charlotte).
```sh
python main.py [PATH_TO_USM_FILE_OR_DIR] [OPTIONS]
```
