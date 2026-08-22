import tomllib

from utils.paths import bundle_root


def read_version() -> str:
    pyproject = tomllib.loads((bundle_root() / "pyproject.toml").read_text("utf-8"))
    return pyproject["project"]["version"]


__version__ = read_version()
