#!/usr/bin/env bash
# 計算する機械（ssh で入れる NVIDIA の GPU の機械。名前は SINDY_HOST）で、学習・評価と結果を返す API を回す。
# 回線に出すのはコード（作業ツリーの差分）と指示と小さい JSON だけで、runs は計算する機械の ~/sindy/runs/ に置いたまま動かさない。
#   SINDY_HOST=<host> ./remote.sh up                  コードを送り、イメージを用意し、API（:8787）を立てる
#   SINDY_HOST=<host> ./remote.sh run <段取りの引数>   コードを送ってから、段取り（orchestrate.ts）を向こうで回す
#   SINDY_HOST=<host> ./remote.sh test                コードを送ってから、Python と TS のテストを向こうで回す
#   SINDY_HOST=<host> ./remote.sh fetch <run の id>   run を 1 つ手元の runs/ へ持ってくる（回線の届かない所で見るとき）
# イメージはベースと道具を digest で、Python の依存を uv.lock で固定した Dockerfile から向こうで作る。API はコードの書き換えで立ち直る（bun --watch）。
set -euo pipefail
shopt -s inherit_errexit # コマンド置換の中の失敗でも止まる

host=${SINDY_HOST:?計算する機械の ssh の名前を SINDY_HOST に入れる}
repo=$(cd "$(dirname "$0")" && pwd)
export DOCKER_HOST="ssh://$host"
name=sindy

# 送るコードの版。書きかけなら差分（未追跡のファイルも含む）のハッシュを足す。段取りが run の meta.json に書く。
version() {
  local head diff
  head=$(git -C "$repo" rev-parse --short=12 HEAD)
  diff=$(cd "$repo" && (git diff HEAD; git ls-files -z --others --exclude-standard | xargs -0 -I{} sh -c 'printf "%s\n" "$1"; cat "$1"' _ {}) | shasum | cut -c1-12)
  if [ -z "$(git -C "$repo" status --porcelain)" ]; then echo "$head"; else echo "$head+$diff"; fi
}

sync() {
  version > "$repo/.code-version"
  ssh "$host" 'mkdir -p ~/sindy/code ~/sindy/runs'
  # 送るのは git が見るものと、版を書いた .code-version だけ（.gitignore で外したものは版に入らないので送らない）
  rsync -a --delete --include .code-version --filter=':- .gitignore' --exclude .git --exclude .venv "$repo/" "$host:sindy/code/"
}

# イメージの名前は依存を決めるファイルのハッシュ。同じなら作り直さない。
image() {
  local tag
  tag=$(cat "$repo/Dockerfile" "$repo/pyproject.toml" "$repo/uv.lock" "$repo/package.json" "$repo/bun.lock" | shasum | cut -c1-12)
  if ! docker image inspect "$name:$tag" > /dev/null 2>&1; then
    tar -C "$repo" -c Dockerfile pyproject.toml uv.lock package.json bun.lock | docker build -t "$name:$tag" - >&2
  fi
  echo "$name:$tag"
}

# 向こうのコンテナで学習か評価が回っているか。
busy() {
  docker exec "$name" ps -eo args 2> /dev/null | grep -q 'neurosurrogate.run\|orchestrate.ts'
}

up() {
  sync
  local tag home
  tag=$(image)
  if [ "$(docker inspect -f '{{.Config.Image}}' "$name" 2> /dev/null)" = "$tag" ]; then
    [ "$(docker inspect -f '{{.State.Running}}' "$name")" = true ] || docker start "$name" > /dev/null
    return
  fi
  if busy; then
    echo "イメージが変わったが、学習か評価が回っているので作り直さない（終わってから up する）" >&2
    exit 1
  fi
  docker rm -f "$name" > /dev/null 2>&1 || true
  home=$(ssh "$host" 'echo $HOME')
  # --init は親を失った run.py を刈り取る（ゾンビの pid を「回っている」と読まないため）
  docker run -d --init --pull never --name "$name" --restart unless-stopped --device nvidia.com/gpu=all -p 8787:8787 \
    -v "$home/sindy/code:/code" -v "$home/sindy/runs:/runs" \
    -e SINDY_HOST="$host" -e SINDY_IMAGE="$(docker image inspect -f '{{.Id}}' "$tag")" "$tag" > /dev/null
  echo "${host}:8787 で API を立てた（${tag}）"
}

case ${1:-} in
  up) up ;;
  run)
    shift
    up
    docker exec "$name" bun evaluation/orchestrate/orchestrate.ts "$@"
    ;;
  test)
    up
    docker exec -e JAX_PLATFORMS=cpu "$name" /venv/bin/python -m pytest -q -p no:cacheprovider
    docker exec "$name" bun test evaluation experiments
    ;;
  fetch)
    run=${2:?run の id（runs/ の下のパス）}
    mkdir -p "$repo/runs/$run"
    rsync -a "$host:sindy/runs/$run/" "$repo/runs/$run/"
    ;;
  *) sed -n '2,8p' "$0" && exit 1 ;;
esac
