#!/usr/bin/env python3
"""
compare-diarizers.py -- diff the speaker structure of the same episodes
transcribed by two different diarizers.

WHY. Sortformer caps at 4 speakers. Measured 19 Aug on MarTech: all 1,150
episodes reported exactly 4, which LOOKED like the cap saturating. It was not.
The four were network announcer, a cold-open montage clip, host and guest --
and on three-part guest episodes host and guest split cleanly (171-183 segments
against 87-100). Sortformer is not merging anyone on this show.

But a panel with three real hosts plus the same production furniture would need
6 or 7, and then it WOULD merge. pyannote has no such cap. This script exists to
find out what pyannote actually does on real episodes BEFORE committing a
2,000-episode re-transcription to it -- Scriberr's pyannote path has never been
run here, so its speed, quality and failure modes are all unmeasured.

WHAT TO LOOK FOR
  n_speakers      pyannote finding MORE than 4 on episodes where Sortformer
                  reported exactly 4 means the cap was binding after all.
                  Finding the SAME structure means it was not, and switching
                  buys nothing on this show.
  top2_share      share of speech held by the two largest speakers. An
                  interview should be high (~90%+). If pyannote drops this
                  sharply it is fragmenting one person across several labels,
                  which is worse, not better.
  null_pct        share of speech with NO speaker assigned. Sortformer leaves
                  ~4.5% on MarTech interviews -- short fragments and sentence
                  continuations in fast exchanges. Text is still indexed, so
                  this costs retrieval nothing, but fewer is better.

USAGE
    python3 scripts/compare-diarizers.py \\
        --a /tmp/diarbench/out-sortformer --a-label sortformer \\
        --b /tmp/diarbench/out-pyannote   --b-label pyannote

Compares only stems present in BOTH directories. Reads exported .json only --
it never touches the API, the audio or scriberr.db.
"""

import argparse
import json
import os
import statistics
import sys


def load(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return (json.load(fh).get("transcript") or {}).get("segments") or []


def profile(segments):
    """Speaker structure of one episode."""
    by_spk = {}
    null_secs = 0.0
    null_segs = 0
    total = 0.0
    for s in segments:
        dur = max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
        total += dur
        spk = s.get("speaker")
        if spk is None:
            null_secs += dur
            null_segs += 1
        else:
            by_spk[spk] = by_spk.get(spk, 0.0) + dur
    ranked = sorted(by_spk.values(), reverse=True)
    voiced = sum(ranked)
    return {
        "n_speakers": len(by_spk),
        "n_segments": len(segments),
        "total_s": round(total, 1),
        "null_segs": null_segs,
        "null_pct": round(100.0 * null_secs / total, 1) if total else 0.0,
        # Share of ATTRIBUTED speech, not of total -- otherwise a diarizer that
        # leaves more unassigned would look artificially better at focusing.
        "top1_share": round(100.0 * ranked[0] / voiced, 1) if voiced else 0.0,
        "top2_share": round(100.0 * sum(ranked[:2]) / voiced, 1) if voiced else 0.0,
        "speaker_secs": [round(v, 1) for v in ranked],
    }


def med(values):
    return round(statistics.median(values), 1) if values else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="directory of exported .json")
    ap.add_argument("--b", required=True, help="directory of exported .json")
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--json", dest="json_out", help="also write the full comparison here")
    args = ap.parse_args()

    for d in (args.a, args.b):
        if not os.path.isdir(d):
            sys.exit(f"ABORT: not a directory: {d}")

    a_stems = {f[:-5] for f in os.listdir(args.a) if f.endswith(".json")}
    b_stems = {f[:-5] for f in os.listdir(args.b) if f.endswith(".json")}
    shared = sorted(a_stems & b_stems)
    if not shared:
        sys.exit("ABORT: no stems present in both directories")
    only_a, only_b = sorted(a_stems - b_stems), sorted(b_stems - a_stems)

    rows = []
    for stem in shared:
        try:
            pa = profile(load(os.path.join(args.a, stem + ".json")))
            pb = profile(load(os.path.join(args.b, stem + ".json")))
        except Exception as exc:
            print(f"  skipping {stem}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rows.append({"stem": stem, args.a_label: pa, args.b_label: pb})

    A, B = args.a_label, args.b_label
    print()
    print(f"{'episode':<44} {A+' spk':>10} {B+' spk':>10} "
          f"{A+' top2':>11} {B+' top2':>11} {A+' null':>11} {B+' null':>11}")
    for r in rows:
        pa, pb = r[A], r[B]
        print(f"{r['stem'][:44]:<44} {pa['n_speakers']:>10} {pb['n_speakers']:>10} "
              f"{pa['top2_share']:>10.1f}% {pb['top2_share']:>10.1f}% "
              f"{pa['null_pct']:>10.1f}% {pb['null_pct']:>10.1f}%")

    print()
    print(f"medians over {len(rows)} episode(s)")
    for label in (A, B):
        print(f"  {label:<12} speakers {med([r[label]['n_speakers'] for r in rows]):>5}  "
              f"top1 {med([r[label]['top1_share'] for r in rows]):>5.1f}%  "
              f"top2 {med([r[label]['top2_share'] for r in rows]):>5.1f}%  "
              f"null {med([r[label]['null_pct'] for r in rows]):>5.1f}%  "
              f"segments {med([r[label]['n_segments'] for r in rows]):>6}")

    more = sum(1 for r in rows if r[B]["n_speakers"] > r[A]["n_speakers"])
    same = sum(1 for r in rows if r[B]["n_speakers"] == r[A]["n_speakers"])
    fewer = len(rows) - more - same
    print()
    print(f"  {B} found MORE speakers than {A} on {more}/{len(rows)} episodes, "
          f"the same on {same}, fewer on {fewer}")
    print()
    print(f"  MORE speakers is only good news if top2_share HOLDS. If {B} finds")
    print(f"  more speakers AND top2_share drops sharply, it is splitting one")
    print(f"  person across several labels rather than separating real voices.")
    print(f"  Read the per-episode speaker_secs in --json before concluding.")

    if only_a or only_b:
        print()
        print(f"NOTE: {len(only_a)} stem(s) only in {A}, {len(only_b)} only in {B} "
              f"-- excluded from every figure above.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"a_label": A, "b_label": B, "episodes": rows},
                      fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
