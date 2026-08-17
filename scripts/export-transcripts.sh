#!/usr/bin/env bash
#
# export-transcripts.sh -- pull finished transcripts out of Scriberr's database
# via the API and write them to the NAS, named after the episode.
#
# WHY THIS EXISTS -- READ BEFORE ASSUMING THERE IS A SIMPLER WAY:
#   Scriberr does NOT write transcript files to disk, despite the profile
#   setting `output_format: "all"`. The per-job directories under
#   data/transcripts/<uuid>/ contain ONLY `transcription.log` (~19 KB each).
#   The transcripts live inside `scriberr.db` -- which is why that file is
#   282 MB for 634 episodes. The UI's download buttons generate txt/json/srt
#   on demand from the database.
#   So there is nothing to rsync. The API is the only export path.
#
# THE JOIN:
#   Transcript directories and job IDs are UUIDs and carry no episode identity.
#   `~/podcasts/.transcribed.tsv`, written by bulk-transcribe.sh, is the join:
#   column 1 is the mp3 filename (which carries date and title), column 2 is
#   the job UUID. This script walks that ledger.
#
# USAGE:
#   read -rsp 'API key: ' APIKEY; echo; export APIKEY
#   ./scripts/export-transcripts.sh
#   LIMIT=3 ./scripts/export-transcripts.sh        # test first
#   FORCE=1 ./scripts/export-transcripts.sh        # re-export existing files
#
# ENV:
#   APIKEY       required
#   SCRIBERR_URL default http://localhost:8080
#   LEDGER       default $HOME/podcasts/.transcribed.tsv
#   OUTDIR       default /storage/nas/ai/scriberr/transcripts
#   LIMIT        0 = all
#   FORCE        1 = overwrite existing output
#
# OUTPUT, per episode, sharing the stem of the source mp3 so transcripts sit in
# the same naming space as the audio and the yt-dlp .info.json sidecars:
#   <stem>.json  full API envelope -- job_id, title, status, created_at, and
#                transcript{segments[], metadata{model_family, model_id,
#                model_version}, model_used, processing_time, language}.
#                THIS IS THE SOURCE OF TRUTH for the retrieval layer: it has
#                float start/end per segment and records which model produced
#                it, which matters the day some episodes get re-run on a
#                better model.
#   <stem>.txt   flattened `[m:ss] speaker_N: text`, one segment per line, for
#                reading and for handing to an LLM. Derived, regenerable.

set -uo pipefail

BASE="${SCRIBERR_URL:-http://localhost:8080}/api/v1"
LEDGER="${LEDGER:-$HOME/podcasts/.transcribed.tsv}"
OUTDIR="${OUTDIR:-/storage/nas/ai/scriberr/transcripts}"
LIMIT="${LIMIT:-0}"
FORCE="${FORCE:-0}"

: "${APIKEY:?export APIKEY before running}"
command -v jq >/dev/null || { echo "jq is required"; exit 1; }
[ -f "$LEDGER" ] || { echo "no ledger at $LEDGER"; exit 1; }

# Refuse to run if the NAS is not actually mounted. `ls` cannot tell a mounted
# share from an empty directory; `mountpoint` can. Without this the script
# would happily write 634 files into the WSL2 vhdx and report success.
if ! mountpoint -q /storage/nas; then
  echo "/storage/nas is not mounted -- refusing to write into a local directory"
  echo "try: ls /storage/nas >/dev/null   (the automount is lazy)"
  exit 1
fi
mkdir -p "$OUTDIR" || exit 1

echo "ledger  $LEDGER ($(wc -l < "$LEDGER") entries)"
echo "outdir  $OUTDIR"
echo

n=0; ok=0; skipped=0; failed=0

while IFS=$'\t' read -r mp3 job status extra; do
  [ -n "${job:-}" ] || continue
  [ "$status" = "completed" ] || { skipped=$((skipped+1)); continue; }

  stem="${mp3%.mp3}"
  out_json="$OUTDIR/$stem.json"
  out_txt="$OUTDIR/$stem.txt"

  if [ "$FORCE" != "1" ] && [ -s "$out_json" ] && [ -s "$out_txt" ]; then
    skipped=$((skipped+1)); continue
  fi

  if [ "$LIMIT" -gt 0 ] && [ "$n" -ge "$LIMIT" ]; then
    echo "LIMIT of $LIMIT reached, stopping."
    break
  fi
  n=$((n+1))

  body=$(curl -sf "$BASE/transcription/$job/transcript" -H "X-API-Key: $APIKEY")
  if [ -z "$body" ] || ! echo "$body" | jq -e '.transcript.segments | length > 0' >/dev/null 2>&1; then
    printf '%-4s FAILED  %s\n' "$n." "$stem"
    failed=$((failed+1))
    continue
  fi

  # Full envelope, pretty-printed so it diffs sanely in git or a review.
  echo "$body" | jq '.' > "$out_json"

  # Flattened reading copy. Timestamps as [m:ss] to match what the UI exports.
  echo "$body" | jq -r '
    .transcript.segments[]
    | "[\(.start | floor | (./60|floor|tostring) + ":" + (.%60|floor|tostring|if length<2 then "0"+. else . end))] \(.speaker // "speaker_?"): \(.text)"
  ' > "$out_txt"

  segs=$(echo "$body" | jq -r '.transcript.segments | length')
  printf '%-4s %-5s segs  %s\n' "$n." "$segs" "$stem"
  ok=$((ok+1))
done < "$LEDGER"

echo
echo "exported $ok   skipped $skipped   failed $failed"
echo "output: $OUTDIR"
