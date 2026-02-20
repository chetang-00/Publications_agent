#!/usr/bin/env sh
# Run a command in frontend/ inside the official Node 22 image.
# Vite 8 / Vitest 5 need Node >= 22.12 (jsdom and react-router >= 22.22); this works on any host Node.
# node_modules lives in a Docker volume so Linux binaries never mix with the host's.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TTY_FLAGS="-i"
[ -t 1 ] && TTY_FLAGS="-it"
exec docker run --rm $TTY_FLAGS \
  -v "$ROOT/frontend":/app \
  -v pa-web-node-modules:/app/node_modules \
  -w /app \
  -e npm_config_update_notifier=false \
  ${NODE_DOCKER_ARGS:-} \
  node:22-alpine "$@"
