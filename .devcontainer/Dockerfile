FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_PROJECT_ENVIRONMENT=/workspaces/.venv \
    UV_LINK_MODE=copy \
    SHELL=/bin/zsh

RUN apt-get update && apt-get install -y \
    git \
    curl \
    zsh \
    && rm -rf /var/lib/apt/lists/* \
    && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

SHELL ["/bin/zsh", "-c"]

WORKDIR /workspaces