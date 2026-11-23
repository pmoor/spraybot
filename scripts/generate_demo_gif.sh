#!/usr/bin/env bash
# Regenerates assets/demo.gif (used in README.md) by rendering a headless
# SprayBot simulation of a red circle to PNG frames and encoding a palette-
# optimized GIF with ffmpeg.
#
# Usage:
#   ./scripts/generate_demo_gif.sh [OUTPUT_GIF]
#
# Optional environment overrides:
#   FPS=15 WIDTH=560 RENDER_RES=1120x630 SPEED=3.5 CAMERA=overview SEED=42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT="${1:-$REPO_ROOT/assets/demo.gif}"
FPS="${FPS:-15}"
WIDTH="${WIDTH:-560}"
RENDER_RES="${RENDER_RES:-1120x630}"
SPEED="${SPEED:-3.5}"
CAMERA="${CAMERA:-overview}"
SEED="${SEED:-42}"

for cmd in cargo ffmpeg; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "error: required command '$cmd' not found in PATH" >&2
        exit 1
    fi
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SVG_PATH="$TMP_DIR/circle.svg"
FRAMES_DIR="$TMP_DIR/frames"
PALETTE_PATH="$TMP_DIR/palette.png"

cat >"$SVG_PATH" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" width="1000" height="600">
  <circle cx="500" cy="300" r="185" fill="none" stroke="#E63946" stroke-width="26"/>
</svg>
EOF

echo "Rendering headless frames (${RENDER_RES} @ ${FPS}fps, speed=${SPEED}x, camera=${CAMERA})..."
(
    cd "$REPO_ROOT"
    cargo run --release -- "$SVG_PATH" \
        --headless \
        --record "$FRAMES_DIR" \
        --fps "$FPS" \
        --resolution "$RENDER_RES" \
        --speed "$SPEED" \
        --camera "$CAMERA" \
        --seed "$SEED"
)

mkdir -p "$(dirname "$OUTPUT")"

echo "Encoding palette-optimized GIF (${WIDTH}px wide @ ${FPS}fps) -> ${OUTPUT}..."
ffmpeg -y -v warning \
    -framerate "$FPS" -start_number 1 -i "$FRAMES_DIR/frame_%06d.png" \
    -vf "scale=${WIDTH}:-1:flags=lanczos,palettegen=stats_mode=diff" \
    -frames:v 1 -update 1 \
    "$PALETTE_PATH"

ffmpeg -y -v warning \
    -framerate "$FPS" -start_number 1 -i "$FRAMES_DIR/frame_%06d.png" \
    -i "$PALETTE_PATH" \
    -lavfi "scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
    -loop 0 \
    "$OUTPUT"

SIZE_BYTES="$(stat -c%s "$OUTPUT")"
SIZE_MIB="$(awk "BEGIN { printf \"%.2f\", $SIZE_BYTES / 1048576 }")"
echo "Done: ${OUTPUT} (${SIZE_MIB} MiB)"
