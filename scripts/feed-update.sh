#!/usr/bin/env bash
#
# feed-update.sh -- the standing feed. For each show in shows.tsv: pull new
# episodes, transcribe them, export the transcripts to the NAS.
#
# Runs hourly from scriberr-feed.timer. Also safe to run by hand.
#
# WHY THIS EXISTS RATHER THAN `scriberr-cli watch`:
#   The watcher covers exactly one of the three steps below. It takes no flags,
#   so it applies one server-wide default profile to every show it uploads, and
#   it writes no ledger line -- which matters because the ledger is the ONLY
#   link between a job UUID and an episode. Scriberr does not write transcript
#   files to disk (they live in scriberr.db), so an export step keyed on that
#   ledger is required either way. Watcher or not, this timer has to exist.
#   Given that, the watcher adds a service and a global settings change and
#   buys nothing.
#
# EVERY STAGE IS IDEMPOTENT, WHICH IS WHAT MAKES AN HOURLY TIMER SAFE:
#   yt-dlp      --download-archive skips anything already fetched. Verified on
#               the Captivate feed: stable UUIDs, not rotating session URLs.
#   transcribe  bulk-transcribe.sh skips any mp3 already in the ledger.
#   export      export-transcripts.sh skips any episode whose .json and .txt
#               both exist and are non-empty.
#   So a typical run does nothing at all, and a missed run costs nothing but
#   time -- the next tick catches up. That is why a gaming rig that is off
#   half the week is an acceptable host for this.
#
# USAGE:
#   export APIKEY=...           # or let systemd supply it, see below
#   ./scripts/feed-update.sh
#   SHOW=dtnt ./scripts/feed-update.sh      # one show only
#   DRY=1 ./scripts/feed-update.sh          # print the plan, touch nothing
#
# ENV:
#   APIKEY       required. Under systemd it comes from EnvironmentFile, never
#                from this file and never from the unit.
#   SCRIBERR_URL default http://localhost:8080
#   SHOWS        default <this script's dir>/shows.tsv
#   SHOW         limit to one slug
#   DRY          1 = report only
#
# ADDING A SHOW: one row in shows.tsv. Its first run is a full backfill of that
#   catalogue, not an increment, and may take hours. That is correct and it is
#   resumable -- systemd will not start a second instance while one is running,
#   and the ledger means a kill costs only the in-flight episode.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHOWS="${SHOWS:-$HERE/shows.tsv}"
BASE="${SCRIBERR_URL:-http://localhost:8080}"
SHOW="${SHOW:-}"
DRY="${DRY:-0}"
LOCK="${LOCK:-/tmp/scriberr-feed.lock}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ABORT: $*"; exit 1; }

# ---------------------------------------------------------------- preflight

[ -f "$SHOWS" ] || die "no show table at $SHOWS"
command -v jq      >/dev/null || die "jq is required"
command -v yt-dlp  >/dev/null || die "yt-dlp is required"
command -v flock   >/dev/null || die "flock is required (util-linux)"
: "${APIKEY:?APIKEY is not set -- export it, or check the EnvironmentFile}"

# Only one run at a time. systemd already refuses to start a second instance of
# the service, but that does not stop a hand-run overlapping a timer-run.
exec 9>"$LOCK" || die "cannot open lock $LOCK"
flock -n 9 || { log "another run holds $LOCK -- exiting"; exit 0; }

# Arm the lazy automount BEFORE anything checks it. /storage/nas is mounted
# noauto,x-systemd.automount on this box, so the NFS mount does not exist until
# something reads the path. export-transcripts.sh guards with `mountpoint -q`,
# which would otherwise be evaluated against an unarmed trigger.
ls /storage/nas >/dev/null 2>&1

# Is Scriberr actually up? On this host dockerd does not run until WSL is
# launched, so "down" is a normal state and needs to read as a clear failure
# rather than a quiet no-op. Exit non-zero so the journal records it.
if ! curl -sf -o /dev/null --max-time 10 "$BASE/api/v1/profiles/" -H "X-API-Key: $APIKEY"; then
  die "Scriberr is not answering at $BASE (is dockerd up? is the API key current?)"
fi

log "feed-update starting  ($BASE)"

# ---------------------------------------------------------------- the loop

shows_run=0; had_error=0

while IFS=$'\t' read -r slug feed audio_dir archive ledger outdir profile_id; do
  case "${slug:-}" in ''|\#*) continue ;; esac
  [ -n "${profile_id:-}" ] || { log "SKIP $slug -- malformed row, expected 7 tab-separated fields"; had_error=1; continue; }
  [ -z "$SHOW" ] || [ "$SHOW" = "$slug" ] || continue

  # The config carries a literal $HOME so it stays free of usernames.
  audio_dir="${audio_dir//\$HOME/$HOME}"
  archive="${archive//\$HOME/$HOME}"
  ledger="${ledger//\$HOME/$HOME}"
  outdir="${outdir//\$HOME/$HOME}"

  shows_run=$((shows_run+1))
  log "--- $slug"

  if [ "$DRY" = "1" ]; then
    log "    feed     $feed"
    log "    audio    $audio_dir"
    log "    archive  $archive  ($([ -f "$archive" ] && wc -l < "$archive" || echo 0) fetched)"
    log "    ledger   $ledger  ($([ -f "$ledger" ] && wc -l < "$ledger" || echo 0) transcribed)"
    log "    outdir   $outdir"
    log "    profile  $profile_id"
    continue
  fi

  mkdir -p "$audio_dir" || { log "    cannot create $audio_dir"; had_error=1; continue; }

  # 1. Pull. --no-progress keeps the journal readable; progress bars in a
  #    non-tty log are thousands of useless lines.
  before=$(find "$audio_dir" -maxdepth 1 -name '*.mp3' | wc -l)
  yt-dlp --restrict-filenames --no-progress \
         --download-archive "$archive" \
         -o "$audio_dir/%(upload_date>%Y-%m-%d)s - %(title)s.%(ext)s" \
         "$feed"
  rc=$?
  after=$(find "$audio_dir" -maxdepth 1 -name '*.mp3' | wc -l)
  new=$((after - before))
  [ "$rc" -eq 0 ] || { log "    yt-dlp exited $rc"; had_error=1; }
  log "    pulled $new new episode(s), $after on disk"

  # 2. Transcribe. Skips everything already in the ledger, so on a quiet hour
  #    this walks 634 lines and does nothing. Cheap.
  if [ "$new" -gt 0 ] || [ ! -f "$ledger" ]; then
    PROFILE_ID="$profile_id" LEDGER="$ledger" "$HERE/bulk-transcribe.sh" "$audio_dir" \
      || { log "    bulk-transcribe.sh exited $?"; had_error=1; }
  else
    log "    nothing new to transcribe"
  fi

  # 3. Export. Runs unconditionally, not only when something was pulled: a run
  #    that transcribed successfully and then failed to export (NAS down, say)
  #    must be recoverable by the next tick without new audio arriving.
  LEDGER="$ledger" OUTDIR="$outdir" "$HERE/export-transcripts.sh" \
    || { log "    export-transcripts.sh exited $?"; had_error=1; }

done < "$SHOWS"

log "feed-update finished  ($shows_run show(s), errors=$had_error)"
exit "$had_error"
