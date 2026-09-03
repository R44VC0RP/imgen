#!/bin/sh
set -eu

REPO_URL="${IMGEN_REPO_URL:-https://github.com/R44VC0RP/imgen.git}"
INSTALL_DIR="${IMGEN_INSTALL_DIR:-$HOME/.local/share/imgen}"
BIN_DIR="${IMGEN_BIN_DIR:-$HOME/.local/bin}"
STAGING="${INSTALL_DIR}.new.$$"

case "$INSTALL_DIR" in
  ""|/|"$HOME")
    echo "imgen installer: unsafe install directory: $INSTALL_DIR" >&2
    exit 1
    ;;
esac

cleanup() {
  rm -rf "$STAGING"
}
trap cleanup EXIT INT TERM

for command in git bun node; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "imgen installer: $command is required" >&2
    exit 1
  }
done

node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 18) ? 0 : 1)' || {
  echo "imgen installer: Node.js 22.18 or newer is required" >&2
  exit 1
}

mkdir -p "$(dirname "$INSTALL_DIR")" "$BIN_DIR"
rm -rf "$STAGING"
git clone --depth 1 "$REPO_URL" "$STAGING"
(
  cd "$STAGING"
  bun install --frozen-lockfile
  bun run build
)

rm -rf "$INSTALL_DIR"
mv "$STAGING" "$INSTALL_DIR"
ln -sf "$INSTALL_DIR/dist/imgen.js" "$BIN_DIR/imgen"
trap - EXIT INT TERM

echo "Installed imgen at $BIN_DIR/imgen"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Add $BIN_DIR to your PATH, then run: imgen login" ;;
esac
