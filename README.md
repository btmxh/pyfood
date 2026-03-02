Development & running
---------------------

This repository used to provide a Docker-based development environment. The
preferred workflow is now Nix flakes. Use direnv or `nix develop` to enter a
nix-managed developer shell which provides Python, maturin and Rust toolchain
without Docker.

Typical steps:

1. direnv allow (or `nix develop`)
2. uv sync       # install python workspace deps via uv
3. maturin develop --manifest-path packages/rsimulator/  # build rsimulator
4. uv run pytest  # run tests

If you must use the old docker flow, the `docker-compose.yaml` file is left
for reference.
