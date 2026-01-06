from pathlib import Path

import orjson
import typer


def calculate_key_from_filename(filename: str) -> int:
    """Calculate encryption key component from filename.
    This is the first part of the key calculation, based on a hash of the filename.
    """
    # Handle special intro files that share the same base name.
    intro_files = [
        "MDAQ001_OPNew_Part1",
        "MDAQ001_OPNew_Part2_PlayerBoy",
        "MDAQ001_OPNew_Part2_PlayerGirl",
    ]
    if filename in intro_files:
        filename = "MDAQ001_OP"

    # Calculate hash: sum = char + 3 * sum for each character.
    sum_val = 0
    for char in filename:
        sum_val = ord(char) + 3 * sum_val

    # Mask to 56 bits (0xFFFFFFFFFFFFFF = 2^56 - 1).
    sum_val &= 0xFFFFFFFFFFFFFF

    # Return sum or default value if zero.
    result = 0x100000000000000
    if sum_val > 0:
        result = sum_val

    return result


def find_key(filename: str) -> int | None:
    """Find encryption key in keys.json database."""
    keys = Path(__file__).parent.parent.joinpath("keys.json")
    if not keys.exists():
        typer.echo("Could not find keys.json at root directory.")
        raise typer.Exit(1)

    data = orjson.loads(keys.read_bytes())

    for version in data["list"]:
        if "videos" in version and filename in version["videos"]:
            key = version.get("key", None)
            return key

        if "videoGroups" in version:
            for group in version["videoGroups"]:
                if filename in group["videos"]:
                    key = group.get("key", None)
                    return key

    return None


def get_decryption_key(filename: str) -> tuple[bytes, bytes] | None:
    """Get complete decryption key for a USM file.

    Combines the filename-based key with key from keys.json to produce
    the final decryption key split into two 4-byte components.
    """
    # Remove extension if present.
    basename = Path(filename).stem
    key1 = calculate_key_from_filename(basename)
    key2 = find_key(basename)
    if key2 is None:
        typer.echo(f"No key found for {basename}.")
        raise typer.Exit(1)

    # finalKey = (key1 + key2) & 0xFFFFFFFFFFFFFF
    final_key = 0x100000000000000
    if ((key1 + key2) & 0xFFFFFFFFFFFFFF) != 0:
        final_key = (key1 + key2) & 0xFFFFFFFFFFFFFF

    # Split 64-bit key into two 32-bit keys (little-endian).
    key_bytes = final_key.to_bytes(8, byteorder="little")
    key1 = key_bytes[:4]
    key2 = key_bytes[4:]

    return key1, key2
