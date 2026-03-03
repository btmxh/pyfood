# pyfood

## Running instructions

This codebase in written in a mix of Rust and Python code.
[PyO3/maturin](https://github.com/PyO3/maturin) is used for Rust-Python FFI.
This repo supports [uv](https://docs.astral.sh/uv/) and traditional venv setups
for Nix.

Run tests:
```sh
# uv
uv run pytest
# venv
source .venv/bin/activate
pip install -e packages/dvrptw packages/rsimulator
pytest
```
