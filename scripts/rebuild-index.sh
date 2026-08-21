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
# TWO COPIES, AND THE LOCAL ONE IS THE HOT PATH.
#   Measured 21 Aug on a 29,722-chunk / 135 MB index -- same machine, same query
#   set, only the file location differing: BM25 fetch p50 **286 ms from the NFS
#   copy against 39 ms from a local copy**, p95 496 ms -> 58 ms. SQLite FTS5 does
#   many small reads and NFS punishes exactly that pattern. So the rerank sidecar
#   reads a WSL-local copy; the NAS copy stays the durable one, the Kopia backup,
#   and the fallback for anything reading over SMB.
#
#   THE STAMP IS WRITTEN ONLY IF BOTH PUBLISHES SUCCEED. Deliberate: a
#   half-published rebuild must look UNCHANGED so the next run retries it. Were
#   the stamp written after the local publish, a failed NAS copy would be
#   recorded as done and the copies would silently diverge -- the exact
#   silent-staleness failure this script exists to prevent.
#
#   Set LOCAL_INDEX= (empty) to publish only to the NAS.
#
# USAGE
#     ./scripts/rebuild-index.sh              # rebuild only if changed
#     ./scripts/rebuild-index.sh --force      # rebuild regardless
#     ./scripts/rebuild-index.sh --dry-run    # report, change nothing
#
# ENV
#     ROOT         default /storage/nas/ai/scriberr/podcasts
#     INDEX        default /storage/nas/ai/scriberr/index/chunks.sqlite
#     LOCAL_INDEX  default ~/.local/share/scriberr/chunks.sqlite
#     PYTHON       default python3

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/storage/nas/ai/scriberr/podcasts}"
INDEX="${INDEX:-/storage/nas/ai/scriberr/index/chunks.sqlite}"
# The copy the rerank sidecar actually queries. Empty disables local publishing.
# `${VAR-default}` not `${VAR:-default}` so LOCAL_INDEX= is respected as "off".
LOCAL_INDEX="${LOCAL_INDEX-$HOME/.local/share/scriberr/chunks.sqlite}"
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

# BUILD TO THE LOCAL PATH FIRST when there is one -- build-index.py writes a
# SQLite file, and writing it locally then copying once beats writing it across
# NFS. 9>&- so the child does not inherit the lock fd and hold it if orphaned.
if [ -n "$LOCAL_INDEX" ]; then
  mkdir -p "$(dirname "$LOCAL_INDEX")" || die "cannot create $(dirname "$LOCAL_INDEX")"
  BUILD_TO="$LOCAL_INDEX.new"
else
  BUILD_TO="$INDEX.new"
fi

if ! "$PYTHON" "$HERE/build-index.py" --root "$ROOT" --index "$BUILD_TO" \
       --build-only 9>&-; then
  die "build-index.py failed -- existing indexes are untouched"
fi
[ -s "$BUILD_TO" ] || die "build produced an empty file -- existing indexes untouched"
bytes=$(stat -c %s "$BUILD_TO")

if [ -n "$LOCAL_INDEX" ]; then
  # Local first: it is the hot path, and rename(2) within a directory is atomic
  # so a reader gets the whole old index or the whole new one, never a partial.
  mv -f "$LOCAL_INDEX.new" "$LOCAL_INDEX" || die "could not publish $LOCAL_INDEX"
  log "published $LOCAL_INDEX  ($(( bytes / 1048576 )) MB)"

  # Then the NAS. A failure here is FATAL AND THE STAMP IS NOT WRITTEN, so the
  # next run rebuilds rather than recording a divergence as done.
  cp -f "$LOCAL_INDEX" "$INDEX.new" \
    || die "could not stage $INDEX.new -- local is current, NAS is STALE, stamp not written"
  nas_bytes=$(stat -c %s "$INDEX.new" 2>/dev/null || echo 0)
  [ "$nas_bytes" = "$bytes" ] \
    || die "NAS copy is $nas_bytes bytes against $bytes locally -- refusing to publish a truncated index"
  mv -f "$INDEX.new" "$INDEX" \
    || die "could not publish $INDEX -- local is current, NAS is STALE, stamp not written"
else
  mv -f "$INDEX.new" "$INDEX" || die "could not publish $INDEX"
fi

# Written LAST, and only once EVERY copy is in place. A half-published rebuild
# must look UNCHANGED so the next run retries it.
printf '%s\n' "$now" > "$STAMP"
# stat, not `du -h`: on this NFS mount du reports raw blocks with no unit, so a
# 40 MB index printed as a bare "512" -- which reads as alarming at 03:30.
log "published $INDEX  ($(( bytes / 1048576 )) MB, $count episodes)"
