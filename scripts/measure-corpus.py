#!/usr/bin/env python3
"""
measure-corpus.py -- read-only measurement pass over the transcript corpus.

WHY THIS EXISTS. The retrieval layer's design questions (chunk boundary,
embedding model, vector store) had no way to be answered, because every
candidate answer was taste. This replaces the guesses with numbers, and
produces a labelled evaluation set as a side effect.

IT WRITES NOTHING EXCEPT ITS OWN OUTPUT FILES. It never touches the
transcripts, the sidecars, the audio, or scriberr.db.

MULTI-SHOW SINCE 2026-08-18. The podcast tree moved onto the NAS and is now
laid out per show:

    <root>/transcripts/<Show>/<stem>.json
    <root>/audio/<Show>/<stem>.info.json      sidecars live WITH the audio

Shows are discovered from the transcripts directory. Every episode, every eval
candidate and every repeated-segment line carries its show, and the aggregates
break down per show as well as overall.

USAGE (on Hermes inside WSL, or anywhere /storage/nas is mounted):
    python3 scripts/measure-corpus.py
    python3 scripts/measure-corpus.py --show DoThisNotThat
    python3 scripts/measure-corpus.py --limit 25          # per show, smoke test

WHAT IT ANSWERS
  1. Do all episodes have publisher "Best Moments" timestamps, or just some?
     Everything downstream depends on this. Measured 17 Aug on Do This NOT
     That: 82.5%, and none at all before mid-2024.
  2. What is the natural chunk size? Measured from the gaps between those
     human-authored topic boundaries, not picked out of the air.
  3. How many episodes are solo monologues? If most are, "chunk on speaker
     turns" collapses to "one chunk per episode" and is not a boundary rule.
     (It is not a boundary rule for a different reason -- see the runbook.)
  4. How much of the corpus is boilerplate and ad read? Found by counting
     repeated segment text, PER SHOW, rather than assuming a fixed intro.
  5. How do three candidate boundary rules compare on words per chunk?

OUTPUTS, written to --out (default <root>/../analysis):
    corpus-measurements.json   per-episode facts, per-show and overall aggregates
    eval-candidates.jsonl      one line per labelled retrieval target
    repeated-segments.txt      per show, where intro/outro/ad reads show up

TOKEN COUNTS ARE ESTIMATES. Deliberately no tiktoken dependency; this reports
words and characters, and a chars/4 estimate clearly labelled as such. Word
counts are what the chunk-size decision actually turns on.
"""

import argparse
import html
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

DEFAULT_ROOT = "/storage/nas/ai/scriberr/podcasts"
DEFAULT_OUT = "/storage/nas/ai/scriberr/analysis"

# Promo terms lifted from a Do This NOT That description block. Used only to
# flag candidate ad reads for eyeballing -- this is a HEURISTIC and is reported
# as one. It is also show-specific and will undercount on any other show; the
# repeated-segment counts are the real instrument.
PROMO_TERMS = [
    "guruconference", "guru events", "eventastic", "newsletter", "subscribe",
    "leave a comment", "follow the show", "link in the show notes",
    "sponsor", "brought to you by", "promo code", "sign up",
]

TAG_RE = re.compile(r"<[^>]+>")
PARA_RE = re.compile(r"</p\s*>|<br\s*/?>", re.I)
MOMENT_RE = re.compile(r"^\(?(\d{1,2}:\d{2}(?::\d{2})?)\)?\s*[-\u2013]?\s*(.+)$")
# Pre-2025 episodes use a different description format: no timestamps, but an
# unordered list of discussion points. Same human-written value, coarser
# location. 25-char floor drops "- " separators and stray dashes.
BULLET_RE = re.compile(r"^[-\u2022*\u2013]\s+(.{25,})$")
BULLET_HEADER_RE = re.compile(
    r"(main\s+discussion\s+points|discussion\s+points|main\s+points|key\s+takeaways|takeaways)", re.I)
# Leading episode markers -- "EP. 20- ", "Ep 206 - ", "#4 ". Stripped so a title
# used as a query does not spend a third of its terms on numbering.
TITLE_LEAD_RE = re.compile(r"^\s*(?:ep(?:isode)?\.?\s*#?\d+|#\d+)\s*[-:\u2013]*\s*", re.I)
# Trailing show-branding suffixes on this publisher's titles.
TITLE_TAIL_RE = re.compile(r"\s*[|l]\s*(jay's scoop|ask us anything).*$", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9']+")
NORM_RE = re.compile(r"[^a-z0-9 ]+")


# --------------------------------------------------------------------- helpers

def strip_html(raw):
    """HTML description -> plain text, one line per paragraph."""
    if not raw:
        return ""
    text = PARA_RE.sub("\n", raw)
    text = TAG_RE.sub("", text)
    text = html.unescape(text).replace("\u3164", " ")
    return "\n".join(line.strip() for line in text.split("\n"))


def to_seconds(stamp):
    p = [int(x) for x in stamp.split(":")]
    return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]


def parse_moments(description):
    """
    Timestamped moments from the description.

    Returns (moments, had_header). Timestamps are counted whether or not a
    'Best Moments' heading is present, and the header flag is reported
    separately, so a publisher using a different layout shows up as a number
    rather than being silently absorbed.
    """
    text = strip_html(description)
    had_header = bool(re.search(r"best\s+moments", text, re.I))
    moments = []
    for line in text.split("\n"):
        m = MOMENT_RE.match(line.strip())
        if not m or not m.group(2).strip():
            continue
        try:
            moments.append((to_seconds(m.group(1)), m.group(2).strip()))
        except ValueError:
            continue
    moments.sort(key=lambda x: x[0])
    return moments, had_header


def parse_bullets(description):
    """
    Untimestamped discussion points -- the fallback ground truth for episodes
    with no Best Moments. Episode-level rather than chunk-level location, which
    is weaker but far from useless: 2023 Do This NOT That episodes run about six
    minutes, so an episode is only three or four chunks.
    """
    text = strip_html(description)
    had_header = bool(BULLET_HEADER_RE.search(text))
    bullets = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or MOMENT_RE.match(line):
            continue
        m = BULLET_RE.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        if "http" in body.lower():          # promo bullets carry bare URLs
            continue
        bullets.append(body)
    return bullets, had_header


def clean_title(raw):
    """
    Episode title -> a usable query string.

    TIER 3 GROUND TRUTH, added 2026-08-19. MarTech produced 1 Best Moment and 0
    discussion bullets across 1,150 episodes -- Art19 descriptions carry neither
    format -- leaving 73% of the index by chunk count unmeasurable. Titles are
    the only human-written per-episode text that show has, and they are
    genuinely descriptive ("Generating Demand Using Email Prospecting and
    Outreach"). Weaker than a timestamped moment: terser, episode-level only,
    and closer to the vague-query category that already accounts for some
    misses. But free, and it generalises to any publisher who writes nothing.

    THIS IS ONLY SAFE BECAUSE TITLES ARE UNINDEXED. build-index.py puts the
    title in the chunk header, which is an UNINDEXED column by default.
    Running with --index-header would put the answer key into the searchable
    text and make every tier-3 score meaningless -- the same contamination the
    Best Moments split avoids, in a new place.
    """
    t = html.unescape(raw or "")
    t = TITLE_LEAD_RE.sub("", t)
    t = TITLE_TAIL_RE.sub("", t)
    t = re.sub(r"[=<>|*_~#]+", " ", t)          # decorative runs: "===>", "<==="
    t = re.sub(r"\s+", " ", t).strip(" -:–")
    return t


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


# ------------------------------------------------------------ show discovery

def discover_shows(root, only=None):
    """
    Find shows from <root>/transcripts/*/ and pair each with its audio
    directory. Returns [(show, transcripts_dir, audio_dir)].

    A show with transcripts but no audio directory is still returned -- the
    sidecars are simply missing, which is a measurable fact, not a reason to
    skip the transcripts.
    """
    troot = os.path.join(root, "transcripts")
    aroot = os.path.join(root, "audio")
    if not os.path.isdir(troot):
        sys.exit(f"ABORT: no transcripts directory at {troot}")
    shows = []
    for name in sorted(os.listdir(troot)):
        tdir = os.path.join(troot, name)
        if not os.path.isdir(tdir):
            continue
        if only and name not in only:
            continue
        shows.append((name, tdir, os.path.join(aroot, name)))
    if not shows:
        sys.exit(f"ABORT: no show directories found under {troot}"
                 + (f" matching {sorted(only)}" if only else ""))
    return shows


def boilerplate_threshold(n_episodes, floor, fraction):
    """
    How many episodes a line must appear in to count as boilerplate.

    PER SHOW, AND PROPORTIONAL. A flat threshold cannot serve a 636-episode
    show and a 30-episode one at the same time: 20 is 3% of the first and 67%
    of the second. The floor stops a tiny show from calling a line boilerplate
    because it happened twice.
    """
    return max(floor, int(math.ceil(fraction * n_episodes)))


# ------------------------------------------------------- candidate boundaries

def chunks_by_moments(segments, moments):
    """Split at publisher-authored topic boundaries. Segments before the first
    moment form their own chunk (usually the intro boilerplate)."""
    if not moments:
        return []
    bounds = [m[0] for m in moments]
    out, cur, idx = [], [], 0
    for seg in segments:
        while idx < len(bounds) and float(seg.get("start", 0)) >= bounds[idx]:
            if cur:
                out.append(cur)
            cur, idx = [], idx + 1
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
    does on a monologue, which is the point -- measured bimodal and useless on
    Do This NOT That (median 115 words, p75 1788, max 3319)."""
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

def measure_episode(show, stem, tpath, spath):
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
    bullets, had_bullet_header = parse_bullets(side.get("description", ""))
    raw_title = tj.get("title") or side.get("title") or ""
    title_query = clean_title(raw_title)
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
        "show": show,
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
        "bullet_header": had_bullet_header,
        "n_bullets": len(bullets),
        "title_query": title_query,
        "n_title_words": words(title_query),
        # A title under 4 words is too vague to grade -- "Ask Us Anything"
        # retrieves nothing meaningful and would only add noise to the score.
        "eval_tier": ("chunk" if moments else "episode" if bullets
                      else "title" if words(title_query) >= 4 else "none"),
        "moment_gaps_s": summarize(gaps),
        "promo_segment_hits": promo_hits,
        "chunks": {
            "by_moments": summarize(chunk_word_counts(chunks_by_moments(segments, moments))),
            "by_window_250": summarize(chunk_word_counts(chunks_by_window(segments, 250))),
            "by_speaker_turn": summarize(chunk_word_counts(chunks_by_speaker(segments))),
        },
    }, segments, moments, bullets


def window_text(segments, t, before=15.0, after=75.0):
    """Transcript text around a moment timestamp, so an eval candidate can be
    eyeballed against what was actually said there."""
    keep = [
        s.get("text", "")
        for s in segments
        if float(s.get("start", 0)) >= t - before and float(s.get("start", 0)) <= t + after
    ]
    return " ".join(keep).strip()


def show_aggregate(episodes):
    """Aggregates over one show's episodes. Same shape for the overall roll-up,
    so per-show and corpus-wide numbers are directly comparable."""
    n = len(episodes) or 1
    with_moments = [e for e in episodes if e["n_moments"] > 0]
    with_bullets = [e for e in episodes if e["n_bullets"] > 0]
    tier2 = [e for e in episodes if e["eval_tier"] == "episode"]
    tier3 = [e for e in episodes if e["eval_tier"] == "title"]
    no_eval = [e for e in episodes if e["eval_tier"] == "none"]
    solo = [e for e in episodes if e["n_speakers"] <= 1]
    all_gaps = [e["moment_gaps_s"]["median"] for e in episodes if e["moment_gaps_s"]["n"]]

    by_year = defaultdict(Counter)
    for e in episodes:
        by_year[(e["upload_date"] or "????")[:4]][e["eval_tier"]] += 1

    def chunk_medians(key):
        return summarize([e["chunks"][key]["median"] for e in episodes if e["chunks"][key]["n"]])

    return {
        "episodes": len(episodes),
        "episodes_missing_sidecar": sum(1 for e in episodes if not e["has_sidecar"]),
        "total_segments": sum(e["n_segments"] for e in episodes),
        "total_words": sum(e["total_words"] for e in episodes),
        "total_est_tokens": sum(e["est_tokens"] for e in episodes),
        "best_moments": {
            "episodes_with_any": len(with_moments),
            "coverage_pct": round(100.0 * len(with_moments) / n, 1),
            "episodes_with_header": sum(1 for e in episodes if e["best_moments_header"]),
            "moments_total": sum(e["n_moments"] for e in episodes),
            "moments_per_episode": summarize([e["n_moments"] for e in with_moments]),
            "median_gap_between_moments_s": summarize(all_gaps),
        },
        "discussion_bullets": {
            "episodes_with_any": len(with_bullets),
            "episodes_with_header": sum(1 for e in episodes if e["bullet_header"]),
            "bullets_total": sum(e["n_bullets"] for e in episodes),
        },
        "eval_coverage": {
            "tier1_chunk_level_episodes": len(with_moments),
            "tier2_episode_level_episodes": len(tier2),
            "tier3_title_level_episodes": len(tier3),
            "no_ground_truth_episodes": len(no_eval),
            "no_ground_truth_stems": [e["stem"] for e in no_eval][:40],
            "combined_coverage_pct": round(100.0 * (len(episodes) - len(no_eval)) / n, 1),
            "by_year": {y: dict(c) for y, c in sorted(by_year.items())},
        },
        "speakers": {
            "solo_episodes": len(solo),
            "solo_pct": round(100.0 * len(solo) / n, 1),
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


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="podcast root holding transcripts/<Show>/ and audio/<Show>/")
    ap.add_argument("--out", default=None, help="default <root>/../analysis")
    ap.add_argument("--show", default=None,
                    help="comma-separated show directory names; default all")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N episodes PER SHOW (smoke test)")
    ap.add_argument("--boiler-floor", type=int, default=10,
                    help="minimum episodes a line must appear in to count as boilerplate")
    ap.add_argument("--boiler-fraction", type=float, default=0.05,
                    help="or this share of the show's episodes, whichever is larger")
    args = ap.parse_args()

    only = {s.strip() for s in args.show.split(",")} if args.show else None
    shows = discover_shows(args.root, only)
    out_dir = args.out or os.path.join(os.path.dirname(args.root.rstrip("/")), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    episodes = []
    eval_rows = []
    unreadable = []
    per_show_repeated = {}
    per_show_pos = {}

    for show, tdir, adir in shows:
        stems = sorted(f[:-5] for f in os.listdir(tdir) if f.endswith(".json"))
        if args.limit:
            stems = stems[: args.limit]
        if not stems:
            print(f"  {show}: no .json transcripts, skipping", file=sys.stderr)
            continue
        if not os.path.isdir(adir):
            print(f"  {show}: WARNING no audio directory at {adir} -- "
                  f"no sidecars, so no eval candidates", file=sys.stderr)

        repeated = Counter()
        repeated_pos = defaultdict(list)
        print(f"  {show}: {len(stems)} episodes", file=sys.stderr)

        for i, stem in enumerate(stems, 1):
            tpath = os.path.join(tdir, stem + ".json")
            spath = os.path.join(adir, stem + ".info.json")
            try:
                ep, segments, moments, bullets = measure_episode(show, stem, tpath, spath)
            except Exception as exc:        # keep going; report at the end
                unreadable.append({"show": show, "stem": stem,
                                   "error": f"{type(exc).__name__}: {exc}"})
                continue
            episodes.append(ep)

            for pos, s in enumerate(segments):
                norm = normalize(s.get("text"))
                if len(norm) >= 15:         # ignore "Yeah." and friends
                    repeated[norm] += 1
                    if len(repeated_pos[norm]) < 200:
                        repeated_pos[norm].append(pos)

            for t, desc in moments:
                eval_rows.append({
                    "tier": "chunk", "show": show, "stem": stem,
                    "episode_number": ep["episode_number"],
                    "upload_date": ep["upload_date"],
                    "t_seconds": t,
                    "publisher_description": desc,
                    "transcript_window": window_text(segments, t),
                })

            # Tier 2 only where there are no timestamps, so an episode carrying
            # both is not counted twice at different precisions.
            if not moments:
                for desc in bullets:
                    eval_rows.append({
                        "tier": "episode", "show": show, "stem": stem,
                        "episode_number": ep["episode_number"],
                        "upload_date": ep["upload_date"],
                        "t_seconds": None,
                        "publisher_description": desc,
                        "transcript_window": None,
                    })

            # Tier 3, last resort: the episode title as the query. Only where
            # the publisher supplied neither timestamps nor bullets. One row per
            # episode, so a show measured this way contributes far fewer queries
            # than one with Best Moments -- that asymmetry is real and should be
            # read as such rather than corrected for.
            if ep["eval_tier"] == "title":
                eval_rows.append({
                    "tier": "title", "show": show, "stem": stem,
                    "episode_number": ep["episode_number"],
                    "upload_date": ep["upload_date"],
                    "t_seconds": None,
                    "publisher_description": ep["title_query"],
                    "transcript_window": None,
                })

            if i % 100 == 0:
                print(f"    ... {i}/{len(stems)}", file=sys.stderr)

        per_show_repeated[show] = repeated
        per_show_pos[show] = repeated_pos

    if not episodes:
        sys.exit("ABORT: no episodes measured")

    # ------------------------------------------------------------ aggregates
    by_show = {}
    for show, _t, _a in shows:
        eps = [e for e in episodes if e["show"] == show]
        if eps:
            by_show[show] = show_aggregate(eps)

    top_repeated = []
    for show, counter in per_show_repeated.items():
        n_eps = len([e for e in episodes if e["show"] == show])
        thresh = boilerplate_threshold(n_eps, args.boiler_floor, args.boiler_fraction)
        if show in by_show:
            by_show[show]["boilerplate_threshold_episodes"] = thresh
        for text, count in counter.most_common(60):
            if count <= 1:
                continue
            positions = per_show_pos[show][text]
            top_repeated.append({
                "show": show,
                "count": count,
                "over_threshold": count >= thresh,
                "median_position": statistics.median(positions) if positions else None,
                "text": text,
            })

    agg = show_aggregate(episodes)
    agg["shows"] = sorted(by_show)
    agg["episodes_unreadable"] = len(unreadable)

    with open(os.path.join(out_dir, "corpus-measurements.json"), "w", encoding="utf-8") as fh:
        json.dump({"aggregate": agg, "by_show": by_show, "unreadable": unreadable,
                   "top_repeated_segments": top_repeated, "episodes": episodes},
                  fh, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "eval-candidates.jsonl"), "w", encoding="utf-8") as fh:
        for row in eval_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "repeated-segments.txt"), "w", encoding="utf-8") as fh:
        fh.write("show\tcount\tover_threshold\tmedian_seg_position\ttext\n")
        for r in top_repeated:
            fh.write(f"{r['show']}\t{r['count']}\t{r['over_threshold']}\t"
                     f"{r['median_position']}\t{r['text']}\n")

    # ---------------------------------------------------------------- report
    print()
    for show in sorted(by_show):
        a = by_show[show]
        b = a["best_moments"]
        ec = a["eval_coverage"]
        print(f"{show}")
        print(f"  episodes          {a['episodes']}  "
              f"({a['episodes_missing_sidecar']} missing a sidecar)")
        print(f"  segments / words  {a['total_segments']:,} / {a['total_words']:,}  "
              f"(~{a['total_est_tokens']:,} est. tokens)")
        print(f"  best moments      {b['episodes_with_any']}/{a['episodes']} "
              f"({b['coverage_pct']}%), {b['moments_total']:,} total, "
              f"median gap {b['median_gap_between_moments_s'].get('median')} s")
        print(f"  eval coverage     chunk {ec['tier1_chunk_level_episodes']}, "
              f"episode {ec['tier2_episode_level_episodes']}, "
              f"title {ec['tier3_title_level_episodes']}, "
              f"none {ec['no_ground_truth_episodes']}  "
              f"({ec['combined_coverage_pct']}%)")
        print(f"  solo episodes     {a['speakers']['solo_episodes']} "
              f"({a['speakers']['solo_pct']}%)   {a['speakers']['distribution']}")
        print(f"  boilerplate       lines in >= {a.get('boilerplate_threshold_episodes')} "
              f"episodes count; {sum(1 for r in top_repeated if r['show'] == show and r['over_threshold'])} "
              f"of the top 60 qualify")
        for k, v in a["candidate_chunk_sizes_words"].items():
            print(f"    {k:<16} median {v.get('median')}  p25 {v.get('p25')}  "
                  f"p75 {v.get('p75')}  max {v.get('max')}")
        print()

    print(f"CORPUS  {agg['episodes']} episodes across {len(by_show)} show(s), "
          f"{agg['total_segments']:,} segments, {agg['total_words']:,} words "
          f"(~{agg['total_est_tokens']:,} est. tokens)")
    print(f"        {len(eval_rows):,} eval candidates "
          f"({sum(1 for r in eval_rows if r['tier'] == 'chunk'):,} chunk-level, "
          f"{sum(1 for r in eval_rows if r['tier'] == 'episode'):,} episode-level, "
          f"{sum(1 for r in eval_rows if r['tier'] == 'title'):,} title-level)")
    print(f"        wrote {out_dir}/"
          "{corpus-measurements.json,eval-candidates.jsonl,repeated-segments.txt}")
    if unreadable:
        print()
        print(f"WARNING: {len(unreadable)} unreadable, first few:")
        for u in unreadable[:5]:
            print(f"  {u['show']}/{u['stem']}: {u['error']}")


if __name__ == "__main__":
    main()
