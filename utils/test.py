from pathlib import Path
import orjson

filename = "Cs_Inazuma_LQ1200905_IttoOniStroy01"

path = Path(__file__).parent.parent.joinpath("keys.json")
with open(path, "rb") as f:
    data = orjson.loads(path.read_bytes())

for version in data["list"]:
    if "videos" in version:
        continue
    if filename in version["videos"]:
        key = version.get("key", 0)
        print(key)