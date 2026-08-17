#!/usr/bin/env bash
#
# bulk-transcribe.sh -- feed a directory of audio files through Scriberr one at a
# time, using a saved transcription profile, and keep a ledger so it is resumable.
#
# Written for the Do This NOT That backfill on Hermes: 634 episodes, ~136.6 h of
# audio, ~14 h at Parakeet's measured 9.62x realtime.
#
# WHY A LOOP AND NOT THE CLI WATCHER:
#   `scriberr-cli watch` takes no flags. It uploads and lets the server decide
#   whether to auto-start and which profile to apply, via the user settings
#   `auto_transcription_enabled` and `default_profile_id`. That is an implicit
#   profile you cannot see per job. This script POSTs the profile's parameters
#   explicitly to /start, so every job carries settings you specified.
#
#   THE WATCHER IS NOT THE RIGHT TOOL FOR THE STANDING FEED EITHER -- an earlier
#   version of this comment said it was, and that was written before we learned
#   Scriberr keeps transcripts in scriberr.db rather than on disk. Rejected
#   again 17 Aug 2026 for three reasons: an export step has to run on a schedule
#   regardless, so the choice was never watcher-vs-timer; the watcher writes no
#   ledger line, and the ledger is the only link between a job UUID and an
#   episode; and one server-wide default profile cannot serve several shows,
#   since Sortformer caps at 4 speakers and a panel show needs pyannote.
#   This script is driven hourly by feed-update.sh. See [[Scriberr Plan]].
#
# USAGE:
#   read -rsp 'API key: ' APIKEY; echo; export APIKEY
#   ./scripts/bulk-transcribe.sh                 # everything not already done
#   LIMIT=3 ./scripts/bulk-transcribe.sh         # first 3 only -- ALWAYS DO THIS FIRST
#   ./scripts/bulk-transcribe.sh /some/other/dir
#
# ENV:
#   APIKEY       required. Create in Settings -> API Keys. Never hardcode it here.
#   SCRIBERR_URL default http://localhost:8080
#   PROFILE_ID   default is the "Podcast Parakeet + Sortformer" profile
#   LIMIT        0 = no limit
#   POLL         seconds between status checks, default 15
#
# The ledger at $LEDGER is the resume mechanism. Delete a line to redo that
# episode. Delete the file to redo everything.

set -uo pipefail

BASE="${SCRIBERR_URL:-http://localhost:8080}/api/v1"
PROFILE_ID="${PROFILE_ID:-56483c7a-76f1-41ea-aa50-03afcf0fd781}"
AUDIO_DIR="${1:-$HOME/podcasts/DoThisNotThat}"
LEDGER="${LEDGER:-$HOME/podcasts/.transcribed.tsv}"
LIMIT="${LIMIT:-0}"
POLL="${POLL:-15}"

: "${APIKEY:?export APIKEY before running -- see usage at the top of this file}"

command -v jq >/dev/null || { echo "jq is required"; exit 1; }
[ -d "$AUDIO_DIR" ] || { echo "no such directory: $AUDIO_DIR"; exit 1; }
touch "$LEDGER"

# Pull the profile's parameters once and reuse for every job. Sending the
# server's own stored object back avoids hand-building a WhisperXParams body
# and drifting from what the UI shows.
PARAMS=$(curl -sf "$BASE/profiles/$PROFILE_ID" -H "X-API-Key: $APIKEY" | jq -c '.parameters')
if [ -z "$PARAMS" ] || [ "$PARAMS" = "null" ]; then
  echo "could not read profile $PROFILE_ID -- check APIKEY and PROFILE_ID"
  exit 1
fi
echo "profile   $(curl -sf "$BASE/profiles/$PROFILE_ID" -H "X-API-Key: $APIKEY" | jq -r '.name')"
echo "ledger    $LEDGER ($(wc -l < "$LEDGER") done)"
echo "source    $AUDIO_DIR"
echo

done_count=0
ok=0; failed=0; skipped=0

for f in "$AUDIO_DIR"/*.mp3; do
  [ -e "$f" ] || { echo "no mp3 files in $AUDIO_DIR"; exit 1; }
  base=$(basename "$f")

  # Resume: ledger holds one tab-separated line per finished episode.
  #
  # DO NOT "SIMPLIFY" THIS BACK INTO A PIPELINE. It was
  #   if cut -f1 "$LEDGER" | grep -qxF "$base"; then
  # and that is a race that silently inverts its own result on a MATCH.
  #
  # `grep -q` exits the instant it matches. If `cut` still has output pending,
  # it dies of SIGPIPE (141), and `set -o pipefail` (line 33) promotes 141 to
  # the pipeline's status, so the `if` reads a successful match as false and
  # the episode is re-transcribed. Finding the entry is what breaks it.
  #
  # WHY THE 634-EPISODE BACKFILL RAN CLEAN AND THIS STILL BIT US, 17 Aug 2026:
  # two conditions must hold together. Column 1 must exceed the 64 KiB pipe
  # buffer, or `cut` finishes writing before `grep` reads a byte and SIGPIPE is
  # impossible; at ~116 bytes per line that happens around line 566. AND the
  # lookup must HIT, because a miss makes `grep` read to EOF and `cut` exit
  # normally. Every backfill lookup was a miss, so the bug could not express
  # itself until the standing feed ran against a full ledger.
  #
  # MEASURED on Hermes with a 642-line ledger: column 1 was 74,357 bytes
  # against a 65,536-byte buffer, and 200 trials of the old pipeline against a
  # known-present name returned 9 false negatives (4.5%). Seven episodes were
  # needlessly re-transcribed in one run, reported as success.
  #
  # One awk process, no pipe, so no pipefail interaction. Exact comparison on
  # tab-separated field 1 -- not a whole-line match, and no regex, so filenames
  # full of punctuation stay safe. Missing or empty ledger yields no match,
  # which is the same behaviour as before. Duplicate lines are harmless: it
  # exits on the first hit.
  if awk -F'\t' -v n="$base" '$1==n {f=1; exit} END {exit f?0:1}' "$LEDGER"; then
    skipped=$((skipped+1))
    continue
  fi

  if [ "$LIMIT" -gt 0 ] && [ "$done_count" -ge "$LIMIT" ]; then
    echo "LIMIT of $LIMIT reached, stopping."
    break
  fi

  # Prefer the real episode title from the yt-dlp sidecar over the sanitised
  # filename -- the sidecar has correct punctuation and the episode number.
  sidecar="${f%.mp3}.info.json"
  if [ -f "$sidecar" ]; then
    title=$(jq -r '.title // empty' "$sidecar")
  fi
  [ -n "${title:-}" ] || title="${base%.mp3}"

  printf '%-4s %s\n' "$((done_count+1))." "$title"

  # Upload without starting, so the profile can be applied explicitly.
  upload=$(curl -sf -X POST "$BASE/transcription/upload" \
            -H "X-API-Key: $APIKEY" \
            -F "audio=@${f}" \
            -F "title=${title}")
  job=$(echo "$upload" | jq -r '.id // empty')
  if [ -z "$job" ]; then
    echo "     UPLOAD FAILED: $(echo "$upload" | head -c 200)"
    failed=$((failed+1)); done_count=$((done_count+1)); title=""
    continue
  fi

  start=$(curl -sf -X POST "$BASE/transcription/$job/start" \
            -H "X-API-Key: $APIKEY" \
            -H 'Content-Type: application/json' \
            -d "$PARAMS")
  if [ -z "$start" ]; then
    echo "     START FAILED for job $job"
    failed=$((failed+1)); done_count=$((done_count+1)); title=""
    continue
  fi

  # Poll. One GPU, one job at a time -- this is deliberate pacing, not politeness.
  t0=$(date +%s)
  while :; do
    sleep "$POLL"
    status=$(curl -sf "$BASE/transcription/$job/status" -H "X-API-Key: $APIKEY" | jq -r '.status // "unknown"')
    case "$status" in
      completed)
        secs=$(( $(date +%s) - t0 ))
        printf '     done in %ss  (%s)\n' "$secs" "$job"
        printf '%s\t%s\t%s\t%s\n' "$base" "$job" "completed" "$secs" >> "$LEDGER"
        ok=$((ok+1)); break ;;
      failed)
        err=$(curl -sf "$BASE/transcription/$job" -H "X-API-Key: $APIKEY" | jq -r '.error_message // "no message"')
        echo "     FAILED: $err"
        printf '%s\t%s\t%s\t%s\n' "$base" "$job" "failed" "$err" >> "$LEDGER"
        failed=$((failed+1)); break ;;
      unknown)
        echo "     lost track of job $job -- status unreadable, moving on"
        failed=$((failed+1)); break ;;
      *) : ;;   # uploaded | pending | processing
    esac
  done

  done_count=$((done_count+1))
  title=""
done

echo
echo "ok $ok   failed $failed   already-done $skipped"
echo "ledger: $LEDGER"
