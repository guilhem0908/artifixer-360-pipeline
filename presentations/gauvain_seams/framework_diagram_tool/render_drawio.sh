#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DRAWIO_VERSION=31.1.5
DRAWIO_APPIMAGE_SHA256=bbc06132f9746e34759ec0db59678e44d012572c5942afcf0684c271d29f5c5c
CACHE_ROOT=${DRAWIO_CACHE_ROOT:-/dev/shm/artifixer-drawio-${DRAWIO_VERSION}}
APPIMAGE="$CACHE_ROOT/drawio-x86_64-${DRAWIO_VERSION}.AppImage"
DRAWIO_BIN="$CACHE_ROOT/squashfs-root/drawio"
XVFB="$CACHE_ROOT/xvfb-root/usr/bin/Xvfb"
PYTHON=${PYTHON:-python3}
SOURCE="$HERE/generated/artifixer360_framework.drawio"
SVG="$HERE/generated/artifixer360_framework.svg"
PDF="$HERE/generated/artifixer360_framework.pdf"
PNG="$HERE/generated/artifixer360_framework.png"

"$HERE/build_framework_drawio.py"

mkdir -p "$CACHE_ROOT"

if [[ ! -x "$DRAWIO_BIN" ]]; then
  if [[ -f "$APPIMAGE" ]] && ! printf '%s  %s\n' "$DRAWIO_APPIMAGE_SHA256" "$APPIMAGE" | sha256sum -c - >/dev/null 2>&1; then
    rm -f -- "$APPIMAGE"
  fi
  if [[ ! -x "$APPIMAGE" ]]; then
    APPIMAGE_PART="${APPIMAGE}.part"
    curl -L --fail --retry 2 \
      "https://github.com/jgraph/drawio-desktop/releases/download/v${DRAWIO_VERSION}/drawio-x86_64-${DRAWIO_VERSION}.AppImage" \
      -o "$APPIMAGE_PART"
    printf '%s  %s\n' "$DRAWIO_APPIMAGE_SHA256" "$APPIMAGE_PART" | sha256sum -c -
    mv -- "$APPIMAGE_PART" "$APPIMAGE"
    chmod +x "$APPIMAGE"
  fi
  (cd "$CACHE_ROOT" && "$APPIMAGE" --appimage-extract >/dev/null)
fi

if [[ ! -x "$XVFB" ]]; then
  mkdir -p "$CACHE_ROOT/xvfb-debs" "$CACHE_ROOT/xvfb-root"
  (
    cd "$CACHE_ROOT/xvfb-debs"
    apt-get download xvfb >/dev/null
    for deb in ./*.deb; do
      dpkg-deb -x "$deb" "$CACHE_ROOT/xvfb-root"
    done
  )
fi

test -x "$DRAWIO_BIN"
test -x "$XVFB"

DISPLAY_NUM=:97
"$XVFB" "$DISPLAY_NUM" -screen 0 1920x1080x24 -nolisten tcp >"$CACHE_ROOT/xvfb.log" 2>&1 &
XVFB_PID=$!
cleanup() {
  kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT
sleep 1

run_drawio() {
  DISPLAY="$DISPLAY_NUM" "$DRAWIO_BIN" --no-sandbox --disable-gpu --disable-update "$@"
}

run_drawio --export --format svg --output "$SVG" --size page --embed-diagram --embed-svg-images --embed-svg-fonts true --theme light "$SOURCE"
run_drawio --export --format pdf --output "$PDF" --size page --theme light "$SOURCE"
run_drawio --export --format png --output "$PNG" --size page --width 1600 --theme light "$SOURCE"

# diagrams.net 31.1.5 rounds this 16:9 page to 1600×901 pixels. Remove the
# single blank export row so the review image is an exact 1600×900 canvas.
"$PYTHON" - "$PNG" <<'PY'
from pathlib import Path
import sys

from PIL import Image

path = Path(sys.argv[1])
with Image.open(path) as image:
    if image.size == (1600, 901):
        normalized = image.crop((0, 0, 1600, 900))
        normalized.save(path, format="PNG", optimize=True)
    elif image.size != (1600, 900):
        raise SystemExit(f"unexpected diagrams.net PNG size: {image.size}")
PY

printf 'source=%s\nsvg=%s\npdf=%s\npng=%s\n' \
  "$SOURCE" "$SVG" "$PDF" "$PNG"
