#!/usr/bin/env bash
# Build the page and publish it to the repository's gh-pages branch (GitHub Pages serves it at
# https://<owner>.github.io/coresample/). The models and Neuroglancer must exist first (npm run
# model; Neuroglancer as .github/workflows/pages.yml builds it into public/ng): the same files that
# workflow checks before it publishes.
# Pushes by URL with the gh credential helper, so no git remote is needed; never force-pushes:
# each deploy is a new commit on top of the branch.
set -euo pipefail
REPO="${REPO:-davidackerman/coresample}"
URL="https://github.com/${REPO}.git"
cd "$(dirname "$0")/.."
for f in models/dinov2-small-112.onnx models/dinov2-small-224.onnx models/dinov2-small-112-tokens.onnx models/morphem-112-tokens.onnx \
         models/pecore-t16-112-tokens.onnx models/NOTICE.txt models/LICENSE-Apache-2.0.txt models/LICENSE-MIT-MorphEm.txt ng/index.html; do
  [ -f "public/$f" ] || { echo "public/$f is missing (models: npm run model; ng/: as pages.yml builds it)"; exit 1; }
done
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
