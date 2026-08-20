#!/usr/bin/env bash
#
# rebuild-index.sh -- refresh the FTS5 search index when the corpus has changed.
#
# WHY THIS EXISTS. build-index.py is manual, so the index is a SNAPSHOT. The
# feed adds episodes hourly and nothing rebuilds. An agent or a CLI query
# answering from a months-old index is worse than no index at all, because the
# answer looks authoritative and nothing in it reveals the staleness.
#
# WHY IT IS NOT A STAGE INSIDE feed-update.sh:
#   - feed-update.sh holds a lock for the whole tick, which during a backfill is
#     40+ minutes. The rebuild has no reason to wait behind transcription.
#   - A backfill changes the corpus EVERY hour, so an inline rebuild would run
#     ~24 times a day to produce an index nobody queried in between.
#   - Separate failure domains: a broken rebuild should not mark the feed
#     failed, and vice versa.
#
# CHANGE DETECTION. Counts transcripts and takes the newest mtime across every
# show, and compares against a stamp file next to the index. Cheap (a find over
# ~2k files), and it catches additions, replacements and deletions. It does NOT
# catch an edit that preserves both count and mtime -- use --force for that.
#
# ATOMIC PUBLISH. Builds to <index>.new and renames into place. rename(2) within
# one directory is atomic, so a reader either gets the whole old index or the
# whole new one, never a half-written file. This matters as soon as anything
# queries the index on a schedule.
#
# USAGE
#     ./scripts/rebuild-index.sh              # rebuild only if changed
#     ./scripts/rebuild-index.sh --force      # rebuild regardless
#     ./scripts/rebuild-index.sh --dry-run    # report, change nothing
#
# ENV
#     ROOT    default /storage/nas/ai/scriberr/podcasts
#     INDEX   default /storage/nas/ai/scriberr/index/chunks.sqlite
#     PYTHON  default python3

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/storage/nas/ai/scriberr/podcasts}"
INDEX="${INDEX:-/storage/nas/ai/scriberr/index/chunks.sqlite}"
PYTHON="${PYTHON:-python3}"
STAMP="$INDEX.stamp"
LOCK="${LOCK:-/tmp/scriberr-rebuild.lock}"

FORCE=0
DRY=0
for a in "$@"; do
  case "$a" in
    --force)   FORCE=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ABORT: $*"; exit 1; }

command -v flock >/dev/null || die "flock is required (util-linux)"
command -v "$PYTHON" >/dev/null || die "$PYTHON not found"

# Its own lock, NOT the feed's -- see the header. `<>` not `>` so a contender
# cannot truncate the holder's PID before flock turns it away.
exec 9<>"$LOCK" || die "cannot open lock $LOCK"
if ! flock -n 9; then
  holder=$(head -1 "$LOCK" 2>/dev/null)
  log "pid ${holder:-unknown} already rebuilding -- exiting"
  exit 0
fi
printf '%s\n' "$$" > "$LOCK"

# Arm the lazy automount and VERIFY it. An unmounted automount point is an
# ordinary empty directory that reads fine, so `ls` succeeding proves nothing --
# and an empty corpus would happily build an EMPTY INDEX over a good one.
ls /storage/nas >/dev/null 2>&1
mountpoint -q /storage/nas || die "/storage/nas is not mounted"

TDIR="$ROOT/transcripts"
[ -d "$TDIR" ] || die "no transcripts directory at $TDIR"

count=$(find "$TDIR" -name '*.json' -type f | wc -l)
newest=$(find "$TDIR" -name '*.json' -type f -printf '%T@\n' 2>/dev/null \
         | sort -n | tail -1 | cut -d. -f1)
now="${count}:${newest:-0}"
was=$(cat "$STAMP" 2>/dev/null || echo "none")

log "corpus $now   stamp $was"

[ "$count" -gt 0 ] || die "no transcripts found -- refusing to build an empty index"

if [ "$DRY" = "1" ]; then
  [ "$now" = "$was" ] && log "unchanged, would skip" || log "changed, would rebuild"
  exit 0
fi

if [ "$now" = "$was" ] && [ "$FORCE" != "1" ]; then
  log "unchanged, skipping (use --force to rebuild anyway)"
  exit 0
fi

log "rebuilding ..."
# --build-only skips the eval set and all scoring. 9>&- so the child does not
# inherit the lock fd and hold it if it is ever orphaned.
if ! "$PYTHON" "$HERE/build-index.py" --root "$ROOT" --index "$INDEX.new" \
       --build-only 9>&-; then
  die "build-index.py failed -- the existing index is untouched"
fi

[ -s "$INDEX.new" ] || die "build produced an empty file -- existing index untouched"

mv -f "$INDEX.new" "$INDEX" || die "could not publish $INDEX.new"
printf '%s\n' "$now" > "$STAMP"
log "published $INDEX  ($(du -h "$INDEX" | cut -f1))"
