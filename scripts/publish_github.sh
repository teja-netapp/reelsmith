#!/usr/bin/env bash
# One-time: turn this folder into a git repo and push it to a new PRIVATE GitHub repository.
# Usage: scripts/publish_github.sh [repo-name]       (default name: reelsmith)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-reelsmith}"

command -v git >/dev/null 2>&1 || { echo "git is missing - run: xcode-select --install"; exit 1; }
if [ -z "$(git config --global user.email || true)" ]; then
  echo "Set your git identity first:"
  echo '  git config --global user.name  "Your Name"'
  echo '  git config --global user.email "you@example.com"'
  exit 1
fi

[ -d .git ] || git init -b main
git add -A
VER="$(sed -nE 's/^__version__ = "(.*)"/\1/p' app/__init__.py)"
git commit -m "Reelsmith v${VER}" >/dev/null 2>&1 && echo "Committed v${VER}" || echo "Nothing new to commit"

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "Installing GitHub CLI…"; brew install gh
fi
# the 'workflow' scope is required to push .github/workflows (the automatic tests)
if gh auth status >/dev/null 2>&1; then
  gh auth status 2>&1 | grep -q "workflow" || gh auth refresh -h github.com -s workflow
else
  gh auth login -h github.com -p https -w -s workflow
fi
gh auth setup-git >/dev/null 2>&1 || true
gh repo create "$NAME" --private --source=. --remote=origin --push
echo
echo "Pushed to: $(gh repo view --json url -q .url)"
echo "On another Mac:  git clone $(gh repo view --json sshUrl -q .sshUrl) && cd $NAME && ./setup.sh"
