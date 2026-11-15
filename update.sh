#!/usr/bin/env bash
# Pull the latest Reelsmith from GitHub and refresh dependencies. Your .env, models and reels are untouched.
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "You have local code changes. Commit or stash them first (git stash), then re-run ./update.sh"; exit 1
fi
OLD="$(git rev-parse --short HEAD)"
git pull --ff-only
NEW="$(git rev-parse --short HEAD)"

if [ "$OLD" = "$NEW" ]; then echo "Already up to date ($NEW)."; else echo "Updated $OLD -> $NEW"; git log --oneline "$OLD..$NEW" | head -20; fi

if [ ! -d .venv ]; then ./setup.sh; exit 0; fi
# shellcheck disable=SC1091
source .venv/bin/activate
if git diff --name-only "$OLD" "$NEW" | grep -qE '^requirements'; then
  echo "Dependencies changed - installing…"; pip install -r requirements.txt
fi
# add any new settings from .env.example to .env without touching your values
if [ -f .env ]; then
  while IFS= read -r line; do
    key="${line%%=*}"
    case "$line" in ''|'#'*) continue;; esac
    grep -qE "^#? ?${key}=" .env || { echo "$line" >> .env; echo "  added new setting $key to .env"; }
  done < .env.example
fi
echo "Done. Restart the app with ./run.sh"
