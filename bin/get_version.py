#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


try:
    from setuptools_scm import get_version
except ModuleNotFoundError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "setuptools_scm"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from setuptools_scm import get_version


def main() -> None:
    root = Path(__file__).parent.parent
    version = get_version(root=root)
    if version is None:
        version = "unknown"
    print(version)


if __name__ == "__main__":
    main()
