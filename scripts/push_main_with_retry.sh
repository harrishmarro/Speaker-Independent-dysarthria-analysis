#!/usr/bin/env bash
# Reconcile a scheduled runner commit with concurrent updates to main.
set -euo pipefail

for attempt in 1 2 3; do
  git -c rebase.autoStash=true pull --rebase origin main
  if git push origin HEAD:main; then
    exit 0
  fi
  echo "main advanced during push (attempt ${attempt}/3)" >&2
done

echo "could not publish the experiment commit after three attempts" >&2
exit 1
