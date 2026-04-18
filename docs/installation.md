# Installation

`pyswarm` can be installed from PyPI or directly from source via GitHub.

---

## [PyPI](https://pypi.org/project/pyswarm)

For using the PyPI package in your project, add it to your configuration file:

=== "pyproject.toml"

    ```toml
    [project.dependencies]
    pyswarm = "*" # (1)!
    ```

    1. Specifying a version is recommended

=== "requirements.txt"

    ```
    pyswarm>=0.1.0
    ```

### pip

=== "Installation for user"

    ```bash
    pip install --upgrade --user pyswarm # (1)!
    ```

    1. You may need to use `pip3` instead of `pip` depending on your Python installation.

=== "Installation in virtual environment"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install --require-virtualenv --upgrade pyswarm # (1)!
    ```

    1. You may need to use `pip3` instead of `pip` depending on your Python installation.

    !!! note
        The command to activate the virtual environment depends on your platform and shell.
        [More info](https://docs.python.org/3/library/venv.html#how-venvs-work)

### uv

=== "Adding to uv project"

    ```bash
    uv add pyswarm
    uv sync
    ```

=== "Installing to uv environment"

    ```bash
    uv venv
    uv pip install pyswarm
    ```

### pipenv

```bash
pipenv install pyswarm
```

### poetry

```bash
poetry add pyswarm
```

### pdm

```bash
pdm add pyswarm
```

### hatch

```bash
hatch add pyswarm
```

---

## [GitHub](https://github.com/eggzec/pyswarm)

Install the latest development version directly from the repository:

```bash
pip install --upgrade "git+https://github.com/eggzec/pyswarm.git#egg=pyswarm"
```

### Building locally

Clone and build from source if you want to modify or test local changes:

```bash
git clone https://github.com/eggzec/pyswarm.git
cd pyswarm
pip install -e .
```

---

## Dependencies

- Python 3.10+
- NumPy
