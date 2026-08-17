#!/usr/bin/env python3
"""
measure-corpus.py -- read-only measurement pass over the transcript corpus.

WHY THIS EXISTS. The retrieval layer has three open questions (chunk boundary,
embedding model, vector store) and no way to answer any of them, because every
candidate answer is currently taste. This script replaces the guesses with
numbers, and produces a labelled evaluation set as a side effect.

IT WRITES NOTHING EXCEPT ITS OWN OUTPUT FILES. It never touches the transcripts,
the sidecars, the audio, or scriberr.db.

WHAT IT ANSWERS
  1. Do all 634 episodes have publisher "Best Moments" timestamps, or just some?
     Everything downstream depends on this. Verified present on Ep. 551 only.
  2. What is the natural chunk size? Measured from the gaps between those
     human-authored topic boundaries, not picked out of the air.
  3. How many episodes are solo monologues? If most are, "chunk on speaker
     turns" collapses to "one chunk per episode" and is not a boundary rule.
  4. How much of the corpus is boilerplate and ad read? Found by counting
     repeated segment text across all episodes rather than assuming the
     five-segment intro claim holds.
  5. How do three candidate boundary rules actually compare on words per chunk?

USAGE (on Hermes, inside WSL, where /storage/nas is mounted):
    python3 scripts/measure-corpus.py
    python3 scripts/measure-corpus.py --limit 25          # quick smoke test
    python3 scripts/measure-corpus.py --out /tmp/scratch  # somewhere else

OUTPUTS, written to --out (default /storage/nas/ai/scriberr/analysis/):
    corpus-measurements.json   per-episode facts plus aggregates
    eval-candidates.jsonl      one line per Best Moment: the publisher's own
                               description of a passage, the timestamp, and the
                               transcript text actually at that timestamp. This
                               is the retrieval evaluation set -- a query with a
                               known correct answer location, written by humans
                               who listened to the show.
    repeated-segments.txt      the most-repeated segment texts, which is where
                               intro boilerplate and recurring ad reads show up

TOKEN COUNTS ARE ESTIMATES. Deliberately no tiktoken dependency; this reports
words and characters, and a chars/4 estimate clearly labelled as such. Word
counts are what the chunk-size decision actually turns on.
"""

import argparse
import html
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

DEFAULT_TRANSCRIPTS = "/storage/nas/ai/scriberr/transcripts"
DEFAULT_AUDIO = "/storage/nas/ai/scriberr/audio"
DEFAULT_OUT = "/storage/nas/ai/scriberr/analysis"

# Promo terms lifted from the Ep. 551 description block. Used only to flag
# candidate ad reads for eyeballing -- this is a HEURISTIC and is reported as
# one. Do not treat the count as ground truth.
PROMO_TERMS = [
    "guruconference", "eventastic", "newsletter", "subscribe",
    "leave a comment", "follow the show", "link in the show notes",
    "sponsor", "brought to you by", "promo code", "sign up",
]

TAG_RE = re.compile(r"<[^>]+>")
PARA_RE = re.compile(r"</p\s*>|<br\s*/?>", re.I)
# (02:03) or (1:02:03) at the start of a line, then the moment description.
MOMENT_RE = re.compile(r"^\(?(\d{1,2}:\d{2}(?::\d{2})?)\)?\s*[-\u2013]?\s*(.+)$")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
NORM_RE = re.compile(r"[^a-z0-9 ]+")


# --------------------------------------------------------------------- helpers

def strip_html(raw):
    """HTML description -> plain text, one line per paragraph."""
    if not raw:
        return ""
    text = PARA_RE.sub("\n", raw)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    # The feed uses U+3164 HANGUL FILLER as a spacer between promo blocks.
    text = text.replace("\u3164", " ")
    return "\n".join(line.strip() for line in text.split("\n"))


def to_seconds(stamp):
    parts = [int(p) for p in stamp.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_moments(description):
    """
    Pull timestamped moments out of the description.

    Returns (moments, had_header) where moments is [(seconds, text)] and
    had_header records whether a 'Best Moments' heading was present. Timestamps
    are counted whether or not the heading exists, so coverage is measured
    honestly rather than assumed from one episode's layout.
    """
    text = strip_html(description)
    had_header = bool(re.search(r"best\s+moments", text, re.I))
    moments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = MOMENT_RE.match(line)
        if not m:
            continue
        desc = m.group(2).strip()
        if not desc:
            continue
        try:
            moments.append((to_seconds(m.group(1)), desc))
        except ValueError:
            continue
    moments.sort(key=lambda x: x[0])
    return moments, had_header


def words(text):
    return len(WORD_RE.findall(text or ""))


def normalize(text):
    return NORM_RE.sub("", (text or "").lower()).strip()


def pct(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 1),
        "p25": round(pct(values, 25), 1),
        "median": round(statistics.median(values), 1),
        "p75": round(pct(values, 75), 1),
        "p90": round(pct(values, 90), 1),
        "max": round(max(values), 1),
        "mean": round(statistics.fmean(values), 1),
    }


# ------------------------------------------------------- candidate boundaries

def chunks_by_moments(segments, moments):
    """Split at publisher-authored topic boundaries. Segments before the first
    moment form their own chunk (that is usually the intro boilerplate)."""
    if not moments:
        return []
    bounds = [m[0] for m in moments]
    out, cur = [], []
    idx = 0
    for seg in segments:
        while idx < len(bounds) and seg.get("start", 0) >= bounds[idx]:
            if cur:
                out.append(cur)
            cur = []
            idx += 1
        cur.append(seg)
    if cur:
        out.append(cur)
    return out


def chunks_by_window(segments, target_words=250, overlap=1):
    """Fixed target size, breaking only on segment boundaries. Parakeet emits
    clean sentences, so a break here never lands mid-sentence."""
    out, cur, count = [], [], 0
    for seg in segments:
        cur.append(seg)
        count += words(seg.get("text"))
        if count >= target_words:
            out.append(cur)
            cur = cur[-overlap:] if overlap else []
            count = sum(words(s.get("text")) for s in cur)
    if cur and len(cur) > overlap:
        out.append(cur)
    return out


def chunks_by_speaker(segments):
    """Group consecutive segments by speaker. Included to demonstrate what it
    does on a monologue, which is the point."""
    out, cur, who = [], [], None
    for seg in segments:
        spk = seg.get("speaker")
        if who is not None and spk != who and cur:
            out.append(cur)
            cur = []
        who = spk
        cur.append(seg)
    if cur:
        out.append(cur)
    return out


def chunk_word_counts(chunks):
    return [sum(words(s.get("text")) for s in c) for c in chunks]


# --------------------------------------------------------------------- per ep

def measure_episode(stem, tpath, spath):
    with open(tpath, encoding="utf-8", errors="replace") as fh:
        tj = json.load(fh)
    side = {}
    if os.path.exists(spath):
        with open(spath, encoding="utf-8", errors="replace") as fh:
            side = json.load(fh)

    segments = (tj.get("transcript") or {}).get("segments") or []
    seg_words = [words(s.get("text")) for s in segments]
    seg_durs = [
        round(float(s.get("end", 0)) - float(s.get("start", 0)), 3)
        for s in segments
        if s.get("end") is not None and s.get("start") is not None
    ]
    speakers = sorted({s.get("speaker") for s in segments if s.get("speaker")})
    moments, had_header = parse_moments(side.get("description", ""))
    duration = side.get("duration") or (segments[-1].get("end") if segments else 0)

    total_words = sum(seg_words)
    total_chars = sum(len(s.get("text") or "") for s in segments)

    promo_hits = 0
    for s in segments:
        low = (s.get("text") or "").lower()
        if any(term in low for term in PROMO_TERMS):
            promo_hits += 1

    gaps = [b[0] - a[0] for a, b in zip(moments, moments[1:])] if len(moments) > 1 else []

    return {
        "stem": stem,
        "episode_number": side.get("episode_number"),
        "upload_date": side.get("upload_date"),
        "title": tj.get("title") or side.get("title"),
        "duration_s": duration,
        "model_used": (tj.get("transcript") or {}).get("model_used"),
        "n_segments": len(segments),
        "n_speakers": len(speakers),
        "speakers": speakers,
        "total_words": total_words,
        "total_chars": total_chars,
        "est_tokens": round(total_chars / 4),
        "words_per_second": round(total_words / duration, 3) if duration else None,
        "seg_words": summarize(seg_words),
        "seg_duration_s": summarize(seg_durs),
        "has_sidecar": bool(side),
        "best_moments_header": had_header,
        "n_moments": len(moments),
        "moment_gaps_s": summarize(gaps),
        "promo_segment_hits": promo_hits,
        "chunks": {
            "by_moments": summarize(chunk_word_counts(chunks_by_moments(segments, moments))),
            "by_window_250": summarize(chunk_word_counts(chunks_by_window(segments, 250))),
            "by_speaker_turn": summarize(chunk_word_counts(chunks_by_speaker(segments))),
        },
    }, segments, moments


def window_text(segments, t, before=15.0, after=75.0):
    """Transcript text around a moment timestamp, so an eval candidate can be
    eyeballed against what was actually said there."""
    keep = [
        s.get("text", "")
        for s in segments
        if float(s.get("start", 0)) >= t - before and float(s.get("start", 0)) <= t + after
    ]
    return " ".join(keep).strip()


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--audio", default=DEFAULT_AUDIO, help="where the .info.json sidecars live")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="stop after N episodes (smoke test)")
    args = ap.parse_args()

    if not os.path.isdir(args.transcripts):
        sys.exit(f"ABORT: no transcript directory at {args.transcripts}")
    os.makedirs(args.out, exist_ok=True)

    stems = sorted(
        f[:-5] for f in os.listdir(args.transcripts)
        if f.endswith(".json")
    )
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        sys.exit(f"ABORT: no .json transcripts found in {args.transcripts}")

    episodes = []
    repeated = Counter()
    repeated_pos = defaultdict(list)
    eval_rows = []
    unreadable = []

    for i, stem in enumerate(stems, 1):
        tpath = os.path.join(args.transcripts, stem + ".json")
        spath = os.path.join(args.audio, stem + ".info.json")
        try:
            ep, segments, moments = measure_episode(stem, tpath, spath)
        except Exception as exc:            # keep going; report at the end
            unreadable.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
            continue
        episodes.append(ep)

        for pos, s in enumerate(segments):
            norm = normalize(s.get("text"))
            if len(norm) >= 15:             # ignore "Yeah." and friends
                repeated[norm] += 1
                if len(repeated_pos[norm]) < 200:
                    repeated_pos[norm].append(pos)

        for t, desc in moments:
            eval_rows.append({
                "stem": stem,
                "episode_number": ep["episode_number"],
                "upload_date": ep["upload_date"],
                "t_seconds": t,
                "publisher_description": desc,
                "transcript_window": window_text(segments, t),
            })

        if i % 50 == 0:
            print(f"  ... {i}/{len(stems)}", file=sys.stderr)

    # ---------------------------------------------------------- aggregates
    n = len(episodes)
    with_moments = [e for e in episodes if e["n_moments"] > 0]
    solo = [e for e in episodes if e["n_speakers"] <= 1]
    all_gaps = []
    for e in episodes:
        if e["moment_gaps_s"]["n"]:
            all_gaps.append(e["moment_gaps_s"]["median"])

    def chunk_medians(key):
        return summarize([e["chunks"][key]["median"] for e in episodes if e["chunks"][key]["n"]])

    agg = {
        "episodes_measured": n,
        "episodes_unreadable": len(unreadable),
        "episodes_missing_sidecar": sum(1 for e in episodes if not e["has_sidecar"]),
        "total_segments": sum(e["n_segments"] for e in episodes),
        "total_words": sum(e["total_words"] for e in episodes),
        "total_est_tokens": sum(e["est_tokens"] for e in episodes),
        "best_moments": {
            "episodes_with_any": len(with_moments),
            "coverage_pct": round(100.0 * len(with_moments) / n, 1) if n else 0,
            "episodes_with_header": sum(1 for e in episodes if e["best_moments_header"]),
            "moments_total": sum(e["n_moments"] for e in episodes),
            "moments_per_episode": summarize([e["n_moments"] for e in with_moments]),
            "median_gap_between_moments_s": summarize(all_gaps),
        },
        "speakers": {
            "solo_episodes": len(solo),
            "solo_pct": round(100.0 * len(solo) / n, 1) if n else 0,
            "distribution": dict(Counter(e["n_speakers"] for e in episodes)),
        },
        "segments": {
            "per_episode": summarize([e["n_segments"] for e in episodes]),
            "words_per_segment_median": summarize([e["seg_words"]["median"] for e in episodes]),
            "seconds_per_segment_median": summarize([e["seg_duration_s"]["median"] for e in episodes]),
        },
        "candidate_chunk_sizes_words": {
            "by_moments": chunk_medians("by_moments"),
            "by_window_250": chunk_medians("by_window_250"),
            "by_speaker_turn": chunk_medians("by_speaker_turn"),
        },
        "promo_heuristic_segment_hits": sum(e["promo_segment_hits"] for e in episodes),
    }

    top_repeated = [
        {
            "count": c,
            "median_position": statistics.median(repeated_pos[t]) if repeated_pos[t] else None,
            "text": t,
        }
        for t, c in repeated.most_common(60)
        if c > 1
    ]

    out_json = os.path.join(args.out, "corpus-measurements.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {"aggregate": agg, "unreadable": unreadable,
             "top_repeated_segments": top_repeated, "episodes": episodes},
            fh, indent=2, ensure_ascii=False,
        )

    out_eval = os.path.join(args.out, "eval-candidates.jsonl")
    with open(out_eval, "w", encoding="utf-8") as fh:
        for row in eval_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_rep = os.path.join(args.out, "repeated-segments.txt")
    with open(out_rep, "w", encoding="utf-8") as fh:
        fh.write("count\tmedian_seg_position\ttext\n")
        for r in top_repeated:
            fh.write(f"{r['count']}\t{r['median_position']}\t{r['text']}\n")

    # ------------------------------------------------------------- report
    b = agg["best_moments"]
    print()
    print(f"episodes            {n} measured, {len(unreadable)} unreadable, "
          f"{agg['episodes_missing_sidecar']} missing a sidecar")
    print(f"segments            {agg['total_segments']:,}")
    print(f"words               {agg['total_words']:,}   (~{agg['total_est_tokens']:,} est. tokens)")
    print()
    print(f"best moments        {b['episodes_with_any']}/{n} episodes ({b['coverage_pct']}%), "
          f"{b['moments_total']:,} total")
    print(f"                    per episode: median {b['moments_per_episode'].get('median')}, "
          f"range {b['moments_per_episode'].get('min')}-{b['moments_per_episode'].get('max')}")
    print(f"                    median gap between them: "
          f"{b['median_gap_between_moments_s'].get('median')} s")
    print()
    print(f"solo episodes       {agg['speakers']['solo_episodes']}/{n} "
          f"({agg['speakers']['solo_pct']}%)   speaker counts: {agg['speakers']['distribution']}")
    print()
    print("median words per chunk, by boundary rule")
    for k, v in agg["candidate_chunk_sizes_words"].items():
        print(f"  {k:<16} median {v.get('median')}  p25 {v.get('p25')}  p75 {v.get('p75')}  max {v.get('max')}")
    print()
    print(f"promo heuristic     {agg['promo_heuristic_segment_hits']:,} segments matched "
          f"a promo term (HEURISTIC, eyeball repeated-segments.txt)")
    print()
    print(f"eval candidates     {len(eval_rows):,} written")
    print(f"wrote               {out_json}")
    print(f"                    {out_eval}")
    print(f"                    {out_rep}")
    if unreadable:
        print()
        print(f"WARNING: {len(unreadable)} unreadable, first few:")
        for u in unreadable[:5]:
            print(f"  {u['stem']}: {u['error']}")


if __name__ == "__main__":
    main()
