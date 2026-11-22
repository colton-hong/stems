#!/bin/sh
# Install a project-local Node.js into .tools/node (git-ignored). Idempotent.
# Downloads the official tarball from nodejs.org and verifies its SHA-256 against the
# release's SHASUMS256.txt before unpacking. Nothing outside this repo is touched.
set -eu

NODE_VERSION="${NODE_VERSION:-v24.22.3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/.tools/node"

if [ -x "$DEST/bin/node" ] && [ "$("$DEST/bin/node" --version)" = "$NODE_VERSION" ]; then
  echo "node $NODE_VERSION already installed at $DEST"
  exit 0
fi

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PLATFORM=darwin-arm64 ;;
  Darwin-x86_64) PLATFORM=darwin-x64 ;;
  Linux-x86_64) PLATFORM=linux-x64 ;;
  Linux-aarch64) PLATFORM=linux-arm64 ;;
  *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

NAME="node-$NODE_VERSION-$PLATFORM"
BASE="https://nodejs.org/dist/$NODE_VERSION"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "downloading $NAME.tar.gz"
curl -fsSL -o "$TMP/$NAME.tar.gz" "$BASE/$NAME.tar.gz"
curl -fsSL -o "$TMP/SHASUMS256.txt" "$BASE/SHASUMS256.txt"

EXPECTED="$(grep " $NAME.tar.gz\$" "$TMP/SHASUMS256.txt" | cut -d' ' -f1)"
ACTUAL="$(shasum -a 256 "$TMP/$NAME.tar.gz" | cut -d' ' -f1)"
if [ -z "$EXPECTED" ] || [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "checksum mismatch for $NAME.tar.gz" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
tar -xzf "$TMP/$NAME.tar.gz" -C "$DEST" --strip-components=1
echo "installed $("$DEST/bin/node" --version) at $DEST"
