import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", "%d-%m-%Y %H:%M:%S"
)

file_handler = logging.FileHandler("build.log", "w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.addFilter(lambda record: record.levelno != logging.ERROR)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)


def run_command(command: str, cwd: str | None = None) -> None:
    if cwd is None:
        logger.warning(
            "No working directory specified. Using current directory."
        )
        cwd = str(Path.cwd())

    log_file_path = Path(cwd) / "build.log"

    logger.info(f"Executing command: '{command}' in '{cwd}'")

    with subprocess.Popen(
        shlex.split(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=dict(**os.environ, PYTHONUNBUFFERED="1"),
        text=True,
    ) as proc:
        with open(log_file_path, "a", encoding="utf-8") as _log_file:
            for line in proc.stdout:  # type: ignore[union-attr]
                logger.debug(line.rstrip())

        rv = proc.wait()

    if rv != 0:
        logger.error(f"Command exited with status {rv}")
        sys.exit(rv)

    logger.info("Command executed successfully.")


def install() -> None:
    run_command("uv pip install . -v")


def wheel() -> None:
    run_command("uv build --wheel -v")


def clean() -> None:
    logger.debug("Starting cleanup ...")

    run_command("uv pip uninstall pyswarm")

    for entry in Path("").iterdir():
        if entry.name in {
            "dist",
            "build",
            "lib",
            ".pytest_cache",
            ".ruff_cache",
        }:
            logger.info(f"Removing '{entry}'")
            shutil.rmtree(entry)
        if entry.name == "bin" and entry.is_dir():
            for bin_entry in entry.iterdir():
                if bin_entry.is_file() and bin_entry.name.startswith("test_"):
                    logger.info(f"Removing '{bin_entry}'")
                    bin_entry.unlink()
        if entry.name.startswith(".mesonpy"):
            logger.info(f"Removing '{entry}'")
            shutil.rmtree(entry)
        if entry.name.endswith("egg-info"):
            logger.info(f"Removing '{entry}'")
            shutil.rmtree(entry)
        if entry.suffix == ".log":
            logger.info(f"Removing '{entry}'")
            entry.unlink()

    logger.info("Finished cleanup.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PySwarm Build Script")
    parser.add_argument(
        "mode",
        help="""Build mode:
        'install' -- Install Python interface (pip install -e . via meson-python)
        'wheel'   -- Build distributable wheel
        'clean'   -- Remove build artefacts""",
        type=str,
        choices=["install", "wheel", "clean"],
    )

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.mode == "install":
        install()
    if args.mode == "wheel":
        wheel()
    if args.mode == "clean":
        clean()


if __name__ == "__main__":
    main()
