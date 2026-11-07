#!/usr/bin/env bash
# Build the page and publish it to the repository's gh-pages branch (GitHub Pages serves it at
# https://<owner>.github.io/coresample/). The model file must exist first (npm run model).
# Pushes by URL with the gh credential helper, so no git remote is needed; never force-pushes:
# each deploy is a new commit on top of the branch.
set -euo pipefail
REPO="${REPO:-davidackerman/coresample}"
URL="https://github.com/${REPO}.git"
cd "$(dirname "$0")/.."
[ -f public/models/dinov2-small-112.onnx ] || { echo "model missing: run npm run model"; exit 1; }
npx vite build
work="$(mktemp -d)"
git_() { git -c credential.helper= -c 'credential.helper=!gh auth git-credential' "$@"; }
if git_ ls-remote --exit-code --heads "$URL" gh-pages >/dev/null 2>&1; then
  git_ clone -q --depth 1 --branch gh-pages "$URL" "$work"
  find "$work" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
else
  git -C "$work" init -q -b gh-pages
fi
cp -r dist/. "$work/"
touch "$work/.nojekyll"
git -C "$work" add -A
git -C "$work" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" \
  commit -q -m "Deploy $(git rev-parse --short HEAD)" || { echo "nothing changed"; exit 0; }
git_ -C "$work" push -q "$URL" gh-pages
echo "pushed gh-pages; the site is https://${REPO%%/*}.github.io/${REPO##*/}/"
