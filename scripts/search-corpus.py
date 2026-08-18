#!/usr/bin/env python3
"""
search-corpus.py -- query the built index and print passages ready to paste
into an AI as context.

This is the USE tool. `build-index.py` builds and scores; it reloads all 635
episodes and rebuilds the index before it will answer a query, which is fine
for evaluation and useless for actually asking a question. This opens the
existing index read-only and answers in milliseconds.

    python3 scripts/search-corpus.py "subject line tips that still work"
    python3 scripts/search-corpus.py "cold email" -k 8 --since 2025
    python3 scripts/search-corpus.py "AI prompting" --format json

DEFAULTS AND WHY
  -k 6            enough context to be useful, short enough to paste
  --max-per-episode 2
                  This show repeats tactics across years, so an unconstrained
                  top-6 often returns the same episode several times. Two per
                  episode buys breadth, which is what "deep context for a
                  marketing idea" wants. Pass 0 to disable.
  --format md     each passage carries its episode title, number, date and
                  timestamp, because the show is TACTICAL: 2023 platform advice
                  is often actively wrong now, and an undated passage reads as
                  current. The date is not decoration, it is the thing that
                  stops stale tactics being quoted as live ones.

DATE FILTERING IS THE MOST USEFUL FLAG HERE. `--since 2025` when you want what
works now; leave it off when you want to see how a tactic has aged.

Query terms are ORed and ranked by bm25. Minimum term length is 2 -- measured
17 Aug 2026, because 3 silently discards "ai" and 4 also loses "seo", "cta"
and "roi".
"""

import argparse
import json
import os
import re
import sqlite3
import sys

DEFAULT_INDEX = "/storage/nas/ai/scriberr/index/chunks.sqlite"
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def fts_expr(text, min_len=2):
    toks = [t for t in WORD_RE.findall((text or "").lower()) if len(t) >= min_len]
    if not toks:
        return None
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks[:60])


def clock(seconds):
    s = int(seconds or 0)
    return f"{s // 60}:{s % 60:02d}"


def pretty_date(raw):
    raw = raw or ""
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}" if len(raw) == 8 else (raw or "undated")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="+", help="what you want to know")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("-k", type=int, default=6, help="passages to return")
    ap.add_argument("--max-per-episode", type=int, default=2, help="0 disables")
    ap.add_argument("--since", help="earliest upload year or YYYYMMDD")
    ap.add_argument("--until", help="latest upload year or YYYYMMDD")
    ap.add_argument("--format", choices=["md", "text", "json"], default="md")
    ap.add_argument("--min-term-len", type=int, default=2)
    args = ap.parse_args()

    if not os.path.exists(args.index):
        sys.exit(f"ABORT: no index at {args.index}\n"
                 f"       build it with: python3 scripts/build-index.py")

    query = " ".join(args.query)
    expr = fts_expr(query, args.min_term_len)
    if not expr:
        sys.exit("ABORT: query has no searchable terms")

    def bound(v, high):
        if not v:
            return None
        return v if len(v) == 8 else (v + ("1231" if high else "0101"))

    lo, hi = bound(args.since, False), bound(args.until, True)

    db = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    sql = ("SELECT text_for_model, header, stem, upload_date, start_s, end_s,"
           " bm25(chunks) AS score FROM chunks WHERE chunks MATCH ?")
    params = [expr]
    if lo:
        sql += " AND upload_date >= ?"
        params.append(lo)
    if hi:
        sql += " AND upload_date <= ?"
        params.append(hi)
    # Over-fetch so the per-episode cap still has k passages to choose from.
    sql += " ORDER BY score LIMIT ?"
    params.append(max(args.k * 8, 50))

    try:
        rows = db.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        sys.exit(f"ABORT: query failed: {exc}")
    finally:
        db.close()

    picked, per_ep = [], {}
    for r in rows:
        stem = r[2]
        if args.max_per_episode and per_ep.get(stem, 0) >= args.max_per_episode:
            continue
        per_ep[stem] = per_ep.get(stem, 0) + 1
        picked.append(r)
        if len(picked) >= args.k:
            break

    if not picked:
        print(f"Nothing matched \"{query}\""
              + (f" in that date range." if (lo or hi) else "."))
        print("Try fewer or more common words, or widen the dates.")
        return

    if args.format == "json":
        print(json.dumps([{
            "header": r[1], "stem": r[2], "upload_date": r[3],
            "start_s": r[4], "end_s": r[5], "score": r[6],
            "text": r[0],
        } for r in picked], indent=2, ensure_ascii=False))
        return

    if args.format == "md":
        print(f"# Podcast context for: {query}\n")
        print(f"{len(picked)} passages from *Do This, NOT That*, retrieved by "
              f"keyword relevance. **Dates matter** -- this show is tactical, "
              f"and platform advice from an older episode may no longer hold.\n")

    for i, (text, header, stem, date, start_s, end_s, score) in enumerate(picked, 1):
        body = text.split("\n", 1)[1] if "\n" in text else text
        if args.format == "md":
            print(f"## {i}. {header}")
            print(f"*{clock(start_s)}-{clock(end_s)}*\n")
            print(body.strip() + "\n")
        else:
            print(f"[{i}] {header}  ({clock(start_s)}-{clock(end_s)})")
            print(body.strip())
            print()

    if args.format != "json":
        years = sorted({(d or "")[:4] for _, _, _, d, _, _, _ in picked if d})
        print(f"---\n{len(picked)} passages, {len(per_ep)} episodes, "
              f"{'-'.join([years[0], years[-1]]) if years else 'undated'}."
              f"  Query: {query}")


if __name__ == "__main__":
    main()
