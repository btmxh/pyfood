FROM ghcr.io/astral-sh/uv:debian-slim

RUN apt update && \
    apt install -y curl build-essential ca-certificates && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"
