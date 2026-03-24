# Installation

## From source

The source files for Regret can be downloaded from the [Github repo](https://github.com/manojmanikandan7/regret).

You can either clone the public repository:

```sh
git clone https://github.com/manojmanikandan7/regret
```

Or download the [tarball](https://github.com/manojmanikandan7/regret/tarball/main):

```sh
curl -OJL https://github.com/manojmanikandan7/regret/tarball/main
```

Once you have a copy of the source, you can install it with:

```sh
cd regret
uv sync
```

## Docs dependencies

To build or serve documentation with zensical and mkdocstrings-python, install the docs group:

```sh
uv sync --group docs
```

Serve docs locally with live reload:

```sh
uv run --group docs zensical serve
```

Build docs output:

```sh
uv run --group docs zensical build --clean
```
