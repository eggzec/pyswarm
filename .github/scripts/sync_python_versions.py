#!/usr/bin/env python3
"""Sync the Python versions declared in pyproject.toml with upstream.

Reads the active (non-EOL) versions from the ACTIVE_VERSIONS environment
variable -- a JSON array of "X.Y" labels as produced by endoflife.date --
and rewrites, in place:

  * project.classifiers, so the "Programming Language :: Python :: ..."
    entries drop versions that have gone end-of-life and pick up newly
    released ones;
  * project.requires-python, raised to the oldest still-supported version
    when the current floor has gone end-of-life.

The managed block always has the same shape:

    Programming Language :: Python :: <MAJOR>
    Programming Language :: Python :: <every supported X.Y, ascending>

Nothing about the major version is hardcoded. <MAJOR> is the newest major
upstream still supports, and it is *replaced* rather than accumulated --
the day a 4.0 ships the bare-major line flips from 3 to 4 on its own, and
again to 5 when that day comes, with no edit to this script:

    Programming Language :: Python :: 4
    Programming Language :: Python :: 3.14
    Programming Language :: Python :: 4.0

Minors are always written in ascending order, newest at the bottom, and a
still-supported older major keeps its minors listed.

The ":: X :: Only" marker is deliberately never emitted, and any existing
one is removed: requires-python already states the supported range, so the
marker is redundant metadata that has to be hand-maintained to stay true.

The requires-python floor is only ever raised, never lowered. A project
that deliberately requires a newer Python than the oldest supported one --
say ">=3.12" while 3.10 is still alive -- keeps its own floor, and only
the versions at or above it are declared.

Formatting is preserved: entries this script does not own keep their
original tomlkit items, so a file that quotes some classifiers with ' and
others with " does not get reflowed, and new entries copy the quote style
of the entries they replace.

Nothing is written when the declaration already matches, so the calling
workflow opens a pull request only on a real change.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import NoReturn

import tomlkit
from tomlkit.items import Array


PYPROJECT = Path("pyproject.toml")

# The three shapes this script owns. Anything else in the classifier list
# -- including the bare "Programming Language :: Python" and other
# languages such as "Programming Language :: Fortran" -- is left alone.
MAJOR = re.compile(r"^Programming Language :: Python :: (\d+)$")
MINOR = re.compile(r"^Programming Language :: Python :: (\d+\.\d+)$")
ONLY = re.compile(r"^Programming Language :: Python :: (\d+) :: Only$")

FLOOR = re.compile(r">=\s*(\d+(?:\.\d+)*)")
LABEL = re.compile(r"\d+\.\d+")


def key(version: str) -> tuple[int, ...]:
    """Order versions numerically rather than lexically.

    Args:
        version: A dotted version such as "3.9" or "3.10".

    Returns:
        The integer components, so 3.9 sorts before 3.10.
    """
    return tuple(int(part) for part in version.split("."))


def emit(name: str, value: str) -> None:
    """Publish a step output for the calling workflow.

    Args:
        name: Output name.
        value: Output value.
    """
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def fail(message: str) -> NoReturn:
    """Annotate the failure for GitHub and stop.

    Args:
        message: What went wrong.

    Raises:
        SystemExit: Always.
    """
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def owns(value: str) -> bool:
    """Report whether this script manages a classifier.

    Args:
        value: A classifier string.

    Returns:
        True when the entry is one this script rewrites.
    """
    return bool(MAJOR.match(value) or MINOR.match(value) or ONLY.match(value))


def is_literal(item: object) -> bool:
    """Report whether an item uses TOML literal (single) quotes.

    Args:
        item: An item from the classifiers array.

    Returns:
        True when the item is written with literal quotes.
    """
    raw = getattr(item, "as_string", None)
    if raw is None:
        return False
    return str(raw()).lstrip().startswith("'")


def read_active(raw: str) -> list[str]:
    """Parse the active version labels supplied by the workflow.

    Args:
        raw: JSON array of version labels.

    Returns:
        The "X.Y" labels, ascending.
    """
    if not raw:
        fail("ACTIVE_VERSIONS is unset")
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"ACTIVE_VERSIONS is not valid JSON: {exc}")

    # endoflife.date labels release lines as "X.Y"; anything else (a
    # codename, a bare major) is not a line this script can declare.
    seen = {str(v) for v in labels if LABEL.fullmatch(str(v))}
    active = sorted(seen, key=key)
    if not active:
        fail("no active X.Y Python versions in ACTIVE_VERSIONS")
    return active


def build_block(supported: list[str]) -> list[str]:
    """Render the managed classifier block.

    The bare-major line tracks the newest supported major and is replaced
    as it advances, so there is only ever one of them.

    Args:
        supported: Supported "X.Y" versions, ascending.

    Returns:
        The classifier strings, in the order they should appear.
    """
    major = max(int(v.split(".")[0]) for v in supported)
    return [f"Programming Language :: Python :: {major}"] + [
        f"Programming Language :: Python :: {v}" for v in supported
    ]


def splice(items: list[object], wanted: list[str]) -> Array:
    """Replace the managed entries, leaving every other item untouched.

    Args:
        items: The existing classifier items.
        wanted: The managed block to write in their place.

    Returns:
        A new multiline array.
    """
    values = [str(item) for item in items]
    owned = [i for i, value in enumerate(values) if owns(value)]

    # Copy the quote style of the entries being replaced so the rewrite
    # does not churn a file that uses literal strings.
    literal = is_literal(items[owned[0]]) if owned else False
    block = [tomlkit.string(v, literal=literal) for v in wanted]

    drop = set(owned)
    kept = [item for i, item in enumerate(items) if i not in drop]
    if owned:
        # Every owned entry sits at or after the first, so the slice
        # before it survives removal unshifted.
        anchor = owned[0]
    else:
        # Nothing declared yet: seed the block after the bare language
        # marker when there is one, otherwise append it.
        marker = "Programming Language :: Python"
        anchor = values.index(marker) + 1 if marker in values else len(kept)

    array = tomlkit.array()
    for item in kept[:anchor] + block + kept[anchor:]:
        array.append(item)
    array.multiline(multiline=True)
    return array


def minors(block: list[str]) -> list[str]:
    """Extract the per-minor versions from a classifier block.

    Args:
        block: Classifier strings.

    Returns:
        The "X.Y" versions they declare.
    """
    return [m.group(1) for m in map(MINOR.match, block) if m]


def bare_major(block: list[str]) -> str | None:
    """Extract the bare-major version from a classifier block.

    Args:
        block: Classifier strings.

    Returns:
        The major, or None when the block declares none.
    """
    found = [m.group(1) for m in map(MAJOR.match, block) if m]
    return found[0] if found else None


def describe(
    before: list[str], wanted: list[str], floors: tuple[str | None, str]
) -> str:
    """Summarise the change for the pull request body.

    Args:
        before: The classifier block as it was.
        wanted: The classifier block as it will be.
        floors: The old and new requires-python floors.

    Returns:
        Markdown describing what moved.
    """
    was_floor, floor = floors
    old, new = minors(before), minors(wanted)
    added = [v for v in new if v not in old]
    dropped = [v for v in old if v not in new]
    was_major, now_major = bare_major(before), bare_major(wanted)
    only = [c for c in before if ONLY.match(c)]

    lines = [
        "The Python versions declared in `pyproject.toml` no longer match",
        "the versions upstream supports, per",
        "[endoflife.date](https://endoflife.date/python).",
        "",
    ]
    if added:
        lines.append(f"- **Added:** {', '.join(added)}")
    if dropped:
        lines.append(f"- **Dropped (end-of-life):** {', '.join(dropped)}")
    if was_major and now_major and was_major != now_major:
        move = f"`Python :: {was_major}` -> `Python :: {now_major}`"
        lines.append(f"- **Major line:** {move}")
    if only:
        lines.append(f"- **Removed redundant marker:** `{only[0]}`")
    if was_floor != floor:
        lines.append(f"- **`requires-python`:** `>={was_floor}` -> `>={floor}`")
    lines += [
        "",
        f"Declared set is now: {', '.join(new)}.",
        "",
        "Opened automatically by the `Python Versions Sync` workflow.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Rewrite pyproject.toml when its declaration has drifted."""
    active = read_active(os.environ.get("ACTIVE_VERSIONS", "").strip())

    if not PYPROJECT.is_file():
        fail("pyproject.toml not found")

    document = tomlkit.parse(PYPROJECT.read_text(encoding="utf-8"))
    project = document.get("project")
    if project is None:
        fail("pyproject.toml has no [project] table")

    requires = str(project.get("requires-python", "")).strip()
    match = FLOOR.search(requires)
    was_floor = match.group(1) if match else None
    floor = max([f for f in (was_floor, active[0]) if f], key=key)
    supported = [v for v in active if key(v) >= key(floor)]
    if not supported:
        fail(f"floor {was_floor} excludes every active version")

    classifiers = project.get("classifiers")
    if classifiers is None:
        fail("[project] has no classifiers list")

    items = list(classifiers)
    before = [str(item) for item in items if owns(str(item))]
    wanted = build_block(supported)

    if before == wanted and was_floor == floor:
        declared = ", ".join(supported)
        print(f"Already in sync: {declared} (requires-python >={floor})")
        emit("changed", "false")
        return

    project["classifiers"] = splice(items, wanted)
    if was_floor != floor:
        project["requires-python"] = f">={floor}"
    PYPROJECT.write_text(tomlkit.dumps(document), encoding="utf-8")

    body = describe(before, wanted, (was_floor, floor))
    body_file = os.environ.get("PR_BODY_FILE")
    if body_file:
        Path(body_file).write_text(body, encoding="utf-8")

    emit("changed", "true")
    sys.stdout.write(body)


if __name__ == "__main__":
    main()
