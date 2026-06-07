#!/usr/bin/env bash
# ============================================================
# run-pipeline.sh — Weekly forum indexing pipeline
#
# 1. Scrape qualifying threads and build vector index
# 2. Build the static site with Vite
# 3. If index.json changed, commit and push to GitHub
# 4. Ping Healthchecks.io dead-man's switch
#
# Intended to run via cron, e.g.:
#   0 0 * * 0  /home/louquillio/projects/chapala-search/scraper/run-pipeline.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="$HOME/projects/chapala-search"
LOCKDIR="/tmp/chapala-search-pipeline.lock"
LOG_FILE="$PROJECT_DIR/scraper/pipeline.log"
HEALTHCHECKS_URL="${HEALTHCHECKS_URL:-}"  # set in environment or leave blank

# --- Lockfile (atomic mkdir) ---
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Lock held — another pipeline is running. Exiting."
  exit 0
fi
trap "rmdir '$LOCKDIR' 2>/dev/null || true" EXIT

cd "$PROJECT_DIR"

exec >> "$LOG_FILE" 2>&1

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "FAIL: $*"; exit 1; }

# --- Step 1: Scrape & build index ---
log "=== Step 1: Scraping forum (--pages=3) ==="
node scraper/build-index.js --pages=3 || fail "Scraper exited with error"

# --- Step 2: Build static site ---
log "=== Step 2: Building static site ==="
npm run build || fail "Vite build failed"

# --- Step 3: Check for changes & deploy ---
log "=== Step 3: Checking for index changes ==="
if git diff --exit-code docs/index.json > /dev/null 2>&1; then
  log "No changes to index.json — skipping deploy."
else
  log "Index.json changed — committing and pushing..."
  git add docs/
  git commit -m "auto: weekly forum index update [$(date '+%Y-%m-%d')]"
  git push origin main || log "WARN: git push failed (no remote configured?)"
  log "Deploy committed and pushed."
fi

# --- Step 4: Healthchecks.io ping ---
if [ -n "$HEALTHCHECKS_URL" ]; then
  log "=== Step 4: Pinging Healthchecks.io ==="
  curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_URL" > /dev/null 2>&1 && \
    log "Healthcheck ping OK" || \
    log "WARN: Healthcheck ping failed"
fi

log "=== Pipeline complete ==="
