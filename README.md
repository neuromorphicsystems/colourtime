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

```sh
python3 -m venv .
source ./bin/activate
pip3 install -U pip event_stream matplotlib maturin numpy pillow
maturin develop -r
```

## Format

```
isort .; black .; pyright .
```
