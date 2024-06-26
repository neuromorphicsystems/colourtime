```sh
pip install colourtime
```

Colourtime converts an Event Stream file (.es) into a PNG image, where time is represented with colour gradients.

The pip package installs a Python library (`import colourtime`) and command-line executable (`colourtime`).

The following command converts an Event Stream file.

```sh
colourtime /path/to/input.es
```

Run `colourtime --help` to list available options.

Check **python/**init**.py** for details on the Python API.

## Build from source

Local build (first run).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install maturin numpy
maturin develop  # or maturin develop --release to build with optimizations
```

Local build (subsequent runs).

```sh
source .venv/bin/activate
maturin develop  # or maturin develop --release to build with optimizations
```

## Format

```
isort .; black .; pyright .
```
