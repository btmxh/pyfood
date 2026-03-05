build-native:
  uv run maturin develop -m packages/rsimulator/Cargo.toml

test:
  uv run pytest
