import re
from pathlib import Path

import typer


class ASS:
    def __init__(self, srt_file: str, lang: str, custom_style: str | None = None):
        self.srt_file = Path(srt_file)
        self.lang = lang
        self.custom_style = custom_style
        self.fontname = "SDK_JP_Web" if lang == "JP" else "SDK_SC_Web"
        self.dialog_lines: list[str] = []

    def parse_srt(self) -> bool:
        with open(self.srt_file, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        i = 0
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue

            # Check if line is a sequence number
            if not lines[i].strip().isdigit():
                i += 1
                continue

            # We need at least 2 more lines (timing + text)
            if i + 2 >= len(lines):
                break

            # Parse timing line (format: 00:00:50,358 --> 00:00:51,225)
            timing_line = lines[i + 1]
            timing_match = re.findall(r"-?\d\d:\d\d:\d\d,\d\d", timing_line)

            # Skip if timing isn't valid
            if len(timing_match) != 2:
                i += 3
                continue

            # Format dialogue line
            dialog = "Dialogue: 0,"

            # Convert timing format from SRT to ASS
            for time_str in timing_match:
                # Remove leading minus, convert comma to period, remove leading zero from hour
                formatted_time = time_str.replace("-0", "0").replace(",", ".")
                if formatted_time.startswith("0"):
                    formatted_time = formatted_time[1:]
                dialog += formatted_time + ","

            # Add subtitle text
            dialog += "Default,,0,0,0,," + lines[i + 2]
            i += 2

            # Check if subtitle text spans two lines
            if (i + 1 < len(lines)) and lines[i + 1].strip():
                # Check that next line isn't a sequence number
                if not lines[i + 1].strip().isdigit():
                    i += 1
                    dialog += "\\n" + lines[i]

            self.dialog_lines.append(dialog)
            i += 1

        if not self.dialog_lines:
            print(f"Warning: {self.srt_file} is empty or has incorrect format")
            return False

        return True

    def convert_to_ass(self, output_dir: str = ".") -> str:
        output_path = Path(output_dir) / "Subs"
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / (self.srt_file.stem + ".ass")

        # Build ASS file content
        ass_content = []

        # Script Info section
        ass_content.append("[Script Info]")
        ass_content.append("; This is an Advanced Sub Station Alpha v4+ script.")
        ass_content.append("ScriptType: v4.00+")
        ass_content.append("Collisions: Normal")
        ass_content.append("ScaledBorderAndShadow: yes")
        ass_content.append("PlayDepth: 0")
        ass_content.append("PlayResX: 384")
        ass_content.append("PlayResY: 288")
        ass_content.append("")

        # Styles section
        ass_content.append("[V4+ Styles]")
        ass_content.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                          "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
                          "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
                          "MarginL, MarginR, MarginV, Encoding")

        if self.custom_style:
            # Replace {fontname} placeholder with actual font
            style_line = self.custom_style.replace("{fontname}", self.fontname)
            ass_content.append(style_line)
        else:
            # Default style matching official style.
            style_params = [
                "Style: Default", # Format: Name
                f"{self.fontname}", # Fontname
                "14.5", # Fontsize
                "&H00FFFFFF", # PrimaryColour
                "&H000000FF", # SecondaryColour
                "&H00000000", # OutlineColour
                "&H00000000", # BackColour
                "0", # Bold
                "0", # Italic
                "0", # Underline
                "0", # StrikeOut
                "100.0", # ScaleX
                "100.0", # ScaleY
                "0.0", # Spacing
                "0.0", # Angle
                "1", # BorderStyle
                "0.1", # Outline
                "0", # Shadow
                "2", # Alignment
                "10", # MarginL
                "10", # MarginR
                "17", # MarginV
                "1", # Encoding
            ]
            style = ",".join(style_params)
            ass_content.append(style)

        ass_content.append("")

        # Events section
        ass_content.append("[Events]")
        ass_content.append("Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text")

        # Add dialogue lines with HTML tag conversion
        for line in self.dialog_lines:
            if line.strip():
                # Convert HTML-like tags to ASS tags
                converted = self._convert_tags(line)
                ass_content.append(converted)

        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_content))

        typer.echo(f"Converted {self.srt_file.name} to ASS format")
        return str(output_file)

    def _convert_tags(self, line: str) -> str:
        """Convert HTML-like tags to ASS formatting tags."""
        # Convert <u>, <b>, <i> to ASS tags
        line = re.sub(r"<([ubi])>", r"{\\$11}", line)
        line = re.sub(r"</([ubi])>", r"{\\$10}", line)

        # Convert <font color="#RRGGBB"> to ASS color tag
        # ASS uses BGR format, so we need to reverse RGB
        line = re.sub(r'<font\s+color="?#(\w{2})(\w{2})(\w{2})"?>', r"{\\c&H$3$2$1&}", line)
        line = re.sub(r"</font>", "", line)

        return line

    @staticmethod
    def find_subtitles(basename: str, subs_folder: str) -> str | None:
        """Find subtitle file matching the given basename.

        Handles special naming conventions and gender-neutral searches for
        Genshin Impact cutscene subtitles.

        Args:
            basename: Base name of the cutscene file (without extension)
            subs_folder: Folder containing subtitle subdirectories (EN, JP, etc.)

        Returns:
            Base name of subtitle file (without language suffix), or None if not found
        """
        # Hardcoded match fixes for known mismatches
        basename_fixes = {
            "Cs_4131904_HaiDaoChuXian_Boy": "Cs_Activity_4001103_Summertime_Boy",
            "Cs_4131904_HaiDaoChuXian_Girl": "Cs_Activity_4001103_Summertime_Girl",
            "Cs_200211_WanYeXianVideo": "Cs_DQAQ200211_WanYeXianVideo",
        }
        basename = basename_fixes.get(basename, basename)

        # Try to find exact match in EN folder (could be any language folder)
        subs_path = Path(subs_folder) / "EN"
        if not subs_path.exists():
            return None

        # Search for exact match
        matches = list(subs_path.glob(f"{basename}_EN.*"))
        if len(matches) == 1:
            # Remove the "_EN" suffix and extension
            return matches[0].stem[:-3]  # Remove last 3 chars (_EN)

        # Try gender-neutral search (remove _Boy/_Girl suffix)
        neutral_basename = basename.replace("_Boy", "").replace("_Girl", "")
        matches = list(subs_path.glob(f"{neutral_basename}_EN.*"))
        if len(matches) == 1:
            return matches[0].stem[:-3]

        return None
