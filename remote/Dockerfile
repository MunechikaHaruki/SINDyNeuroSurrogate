# 計算する機械（NVIDIA の GPU）で学習・評価・測定と、結果を返す API を回すイメージ。
# ベースと道具は digest で、Python とその依存は uv.lock で固定する（lock が pyproject と合わなければ作らない）。TS（段取りと API）は npm の依存を持たない。
# コードは焼き込まず、送った remote/・shared/・resolved/ を /code に、runs を /runs に載せる（../remote.sh）。

# CUDA 12.8 の devel は ncu（Nsight Compute）を持つ
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
COPY --from=ghcr.io/astral-sh/uv:0.12.22@sha256:f513a91fc62fe7c17567eee97230dd198e43edb8a9fbecca843714a4358fe1bc /uv /usr/local/bin/uv
COPY --from=oven/bun:1.4.2@sha256:9114c058aeae42162ee16dd5084b95fe9473970bb6bcb5b232ab1630f0546895 /usr/local/bin/bun /usr/local/bin/bun

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/venv \
    UV_LINK_MODE=copy \
    UV_LOCKED=1
WORKDIR /deps
COPY pyproject.toml uv.lock ./
RUN uv python install 3.13.5 && uv sync --python 3.13.5 --no-install-project --extra cuda

# 段取りは venv の python を起こす。JAX は VRAM を先取りせず（使った分を測るため）、GPU の演算は決定的にする
ENV SINDY_PYTHON=/venv/bin/python \
    SINDY_RUNS=/runs \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_FLAGS=--xla_gpu_deterministic_ops=true
WORKDIR /code/remote
EXPOSE 8787
CMD ["bun", "--watch", "serve/serve.ts"]
