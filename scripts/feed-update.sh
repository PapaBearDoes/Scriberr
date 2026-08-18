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
#   FULL=1 ./scripts/feed-update.sh         # full feed scan, see --break-on-existing below
#
# ENV:
#   APIKEY       required. Under systemd it comes from EnvironmentFile, never
#                from this file and never from the unit.
#   SCRIBERR_URL default http://localhost:8080
#   SHOWS        default <this script's dir>/shows.tsv
#   SHOW         limit to one slug
#   DRY          1 = report only
#   FULL         1 = scan the whole feed instead of stopping at the first
#                episode already in the archive. Use after a failed download.
#
#                *** A NEW SHOW'S BACKFILL MUST RUN WITH FULL=1. ***
#                Feeds are newest-first, so an interrupted backfill leaves the
#                NEWEST episodes in the archive. Without FULL, the next run hits
#                item 1, finds it archived, breaks immediately, and the backfill
#                never resumes. Keep re-running with FULL=1 until the episode
#                count stops growing, then let the hourly timer take over.
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
FULL="${FULL:-0}"
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
#
# `<>` NOT `>`: a plain `>` truncates on open, so a contender would wipe the
# holder's PID out of the file before flock even told it to back off.
exec 9<>"$LOCK" || die "cannot open lock $LOCK"

if ! flock -n 9; then
  # A bare "another run holds the lock" line is useless: it is identical whether
  # a healthy run is transcribing or something wedged an hour ago and never let
  # go. Report WHO and for HOW LONG so the skip is actionable. Seen 17 Aug 2026,
  # where two starts logged that line and systemd reported both as success.
  holder=$(head -1 "$LOCK" 2>/dev/null)
  if [ -n "${holder:-}" ] && [ -z "${holder//[0-9]/}" ] && kill -0 "$holder" 2>/dev/null; then
    age=$(ps -o etime= -p "$holder" 2>/dev/null | tr -d ' ')
    log "pid $holder has held $LOCK for ${age:-unknown} -- exiting"
  else
    log "$LOCK is held but the holder is unrecorded or gone -- exiting"
  fi
  exit 0
fi

# Record the holder for the message above. Safe to truncate: we hold the lock.
printf '%s\n' "$$" > "$LOCK"

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

while IFS=$'\t' read -r slug feed audio_dir archive ledger outdir profile_id ytflags_extra; do
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
    log "    ytflags  ${ytflags_extra:-<none>}"
    [ -s "$archive" ] || log "    NOTE: empty archive -- this is a BACKFILL, run it with FULL=1"
    continue
  fi

  mkdir -p "$audio_dir" || { log "    cannot create $audio_dir"; had_error=1; continue; }

  # 1. Pull. --no-progress keeps the journal readable; progress bars in a
  #    non-tty log are thousands of useless lines.
  #
  #    --break-on-existing MATTERS MORE THAN IT LOOKS. yt-dlp checks the archive
  #    only AFTER resolving each item's real media URL, which costs two webpage
  #    fetches per episode against Spreaker and Captivate. Without it a quiet
  #    hourly run makes ~1,270 requests to discover that nothing changed --
  #    ~30,000 a day to a publisher's CDN. Measured 17 Aug 2026: ~1.5 items/sec,
  #    about seven minutes for a full pass. The feed is newest-first, so
  #    breaking at the first archived item costs three requests instead.
  #
  #    THE TRADE-OFF: an episode that fails to download is never written to the
  #    archive, and newer episodes above it will stop the scan before yt-dlp
  #    reaches it again. That gap is NOT silent -- the failure makes yt-dlp exit
  #    non-zero, this script logs it and exits non-zero, and systemd marks the
  #    service failed. Recover with:  FULL=1 ./scripts/feed-update.sh
  before=$(find "$audio_dir" -maxdepth 1 -name '*.mp3' | wc -l)
  ytflags=(--restrict-filenames --no-progress --download-archive "$archive")
  [ "$FULL" = "1" ] || ytflags+=(--break-on-existing)
  # Per-show extra flags, field 8. Deliberately UNQUOTED so multiple flags word
  # split. Needed because yt-dlp picks a site-specific extractor by URL and some
  # of them return an empty playlist for a plain RSS feed -- Art19 does exactly
  # that, reporting `Downloading 0 items` with no error. Captivate has no
  # dedicated extractor so it falls through to `generic` and just works.
  # shellcheck disable=SC2086
  [ -z "${ytflags_extra:-}" ] || ytflags+=($ytflags_extra)
  yt-dlp "${ytflags[@]}" \
         -o "$audio_dir/%(upload_date>%Y-%m-%d)s - %(title)s.%(ext)s" \
         "$feed" 9>&-
  rc=$?
  after=$(find "$audio_dir" -maxdepth 1 -name '*.mp3' | wc -l)
  new=$((after - before))
  # 101 is yt-dlp's "stopped early because of --break-on-existing", which is the
  # expected outcome of every quiet run. Treating it as failure would mark the
  # service failed hourly and bury real errors.
  [ "$rc" -eq 0 ] || [ "$rc" -eq 101 ] || { log "    yt-dlp exited $rc"; had_error=1; }
  log "    pulled $new new episode(s), $after on disk"

  # 2. Transcribe. UNCONDITIONAL, like the export below. An earlier version
  #    gated this on `new -gt 0`, which had the same recovery hole the export
  #    stage was deliberately written to avoid: audio pulled by a run that died
  #    before transcribing would never be retried until some unrelated episode
  #    happened to arrive. It also meant the ledger skip check never executed on
  #    a quiet hour, which is exactly when you want it exercised -- it is why a
  #    broken skip check went untested on 17 Aug 2026.
  #    Cost of running it every tick: one awk call per episode and two profile
  #    curls, then "already-done N". Cheap.
  PROFILE_ID="$profile_id" LEDGER="$ledger" "$HERE/bulk-transcribe.sh" "$audio_dir" 9>&- \
    || { log "    bulk-transcribe.sh exited $?"; had_error=1; }

  # 3. Export. Also unconditional: a run that transcribed successfully and then
  #    failed to export (NAS down, say) must be recoverable by the next tick
  #    without new audio arriving.
  LEDGER="$ledger" OUTDIR="$outdir" "$HERE/export-transcripts.sh" 9>&- \
    || { log "    export-transcripts.sh exited $?"; had_error=1; }

done < "$SHOWS"

log "feed-update finished  ($shows_run show(s), errors=$had_error)"
exit "$had_error"
