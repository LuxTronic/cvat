#!/bin/sh
# /models is a mounted volume. An older sidecar ran as root, so files already
# on that volume stay root-owned until this chown. The server then runs as
# uid 1000, the same identity yolov8-cls uses.
set -eu
if [ "$(id -u)" -eq 0 ]; then
    mkdir -p /models
    chown -R 1000:1000 /models
    exec setpriv --reuid=1000 --regid=1000 --clear-groups --inh-caps=-all "$@"
fi
exec "$@"
