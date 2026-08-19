#!/usr/bin/env python3
"""
search-corpus.py -- query the built index and print passages ready to paste
into an AI as context.

This is the USE tool. `build-index.py` builds and scores; it reloads the whole
corpus and rebuilds the index before it will answer a query, which is fine for
evaluation and useless for actually asking a question. This opens the existing
index read-only and answers in milliseconds.

    python3 scripts/search-corpus.py "subject line tips that still work"
    python3 scripts/search-corpus.py "cold email" -k 8 --since 2025
    python3 scripts/search-corpus.py "attribution" --show MarTech
    python3 scripts/search-corpus.py --shows          # what is indexed
    python3 scripts/search-corpus.py "AI prompting" --format json

DEFAULTS AND WHY
  -k 6            enough context to be useful, short enough to paste
  --max-per-episode 2
                  These shows repeat tactics across years, so an unconstrained
                  top-6 often returns the same episode several times. Two per
                  episode buys breadth, which is what "deep context for a
                  marketing idea" wants. Pass 0 to disable.
  --max-per-show 0
                  OFF by default, unlike the per-episode cap. A passage from
                  any show is equally valid, so relevance should decide -- use
                  --show to scope deliberately rather than capping blindly.
                  But watch it: one show can outnumber another 50:1 in the
                  index, and BM25 has no idea that matters. If results start
                  coming back monotonously from one source, this is the lever.
                  Setting it switches to one query PER SHOW merged by score,
                  because a cap applied to a single global result set cannot
                  diversify -- it can only return fewer passages.
  --format md     each passage carries its SHOW, episode title, number, date
                  and timestamp. These are TACTICAL podcasts: 2018 martech
                  advice predates GA4, iOS ATT and LLMs, and an undated passage
                  reads as current. Show and date are not decoration -- they are
                  what stops stale tactics being quoted as live ones.

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
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def rerank(rows, query, model_name, fp16):
    """
    Reorder BM25 candidates with a cross-encoder. Measured 19 Aug on 500 held-out
    queries: recall@1 0.506 -> 0.628, recall@5 0.736 -> 0.804, MRR 0.606 -> 0.708.
    The gain concentrates at the TOP of the list, which is what matters for a
    tool that hands six passages to a model.

    Scores `body` rather than `text_for_model`, matching how it was evaluated --
    text_for_model carries the header, and the header carries the episode title.

    OFF BY DEFAULT. Without it this tool has no dependencies and answers in
    milliseconds; with it, a cold start loads ~2.3 GB of model.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("--rerank needs sentence-transformers:\n"
              "  ~/venvs/rerank/bin/pip install sentence-transformers\n"
              "  then run this script with ~/venvs/rerank/bin/python",
              file=sys.stderr)
        return rows
    device = "cpu"
    kwargs = {}
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            if fp16:
                kwargs = {"model_kwargs": {"torch_dtype": torch.float16}}
    except ImportError:
        pass
    print(f"reranking {len(rows)} candidates with {model_name} on {device} ...",
          file=sys.stderr)
    try:
        model = CrossEncoder(model_name, device=device, max_length=512, **kwargs)
    except Exception as exc:
        print(f"  fp16 unavailable ({type(exc).__name__}), using fp32", file=sys.stderr)
        model = CrossEncoder(model_name, device=device, max_length=512)
    scores = model.predict([(query, r[8]) for r in rows], batch_size=16,
                           show_progress_bar=False)
    return [rows[j] for j in sorted(range(len(rows)), key=lambda j: -scores[j])]


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
    ap.add_argument("query", nargs="*", help="what you want to know")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("-k", type=int, default=6, help="passages to return")
    ap.add_argument("--show", default=None,
                    help="comma-separated show names; default all")
    ap.add_argument("--shows", action="store_true",
                    help="list indexed shows with chunk counts and date ranges")
    ap.add_argument("--max-per-episode", type=int, default=2, help="0 disables")
    ap.add_argument("--max-per-show", type=int, default=0, help="0 disables")
    ap.add_argument("--since", help="earliest upload year or YYYYMMDD")
    ap.add_argument("--until", help="latest upload year or YYYYMMDD")
    ap.add_argument("--format", choices=["md", "text", "json"], default="md")
    ap.add_argument("--min-term-len", type=int, default=2)
    ap.add_argument("--rerank", action="store_true",
                    help="reorder BM25 candidates with a cross-encoder (needs the venv)")
    ap.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    ap.add_argument("--rerank-depth", type=int, default=50,
                    help="candidates to rerank; 100 raises the ceiling ~2 points "
                         "and doubles latency")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false",
                    help="full precision; fp16 halved latency with identical results")
    args = ap.parse_args()

    # `nargs="*"` rather than "+" so a bare run prints the full help instead of
    # `error: the following arguments are required: query`, which is the least
    # useful thing to show someone opening the tool for the first time.
    if not args.query and not args.shows:
        ap.print_help()
        return

    if not os.path.exists(args.index):
        sys.exit(f"ABORT: no index at {args.index}\n"
                 f"       build it with: python3 scripts/build-index.py")

    if args.shows:
        db = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
        rows = db.execute(
            "SELECT show, COUNT(*), COUNT(DISTINCT stem),"
            " MIN(upload_date), MAX(upload_date)"
            " FROM chunks GROUP BY show ORDER BY show").fetchall()
        db.close()
        print(f"{'show':<20} {'chunks':>8} {'episodes':>9}  dates")
        for show, chunks, eps, lo_d, hi_d in rows:
            print(f"{show:<20} {chunks:>8} {eps:>9}  "
                  f"{pretty_date(lo_d)} to {pretty_date(hi_d)}")
        return

    shows = [s.strip() for s in args.show.split(",")] if args.show else None

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

    def fetch(show_filter, limit):
        sql = ("SELECT text_for_model, header, stem, upload_date, start_s, end_s,"
               " bm25(chunks) AS score, show, body FROM chunks WHERE chunks MATCH ?")
        p = [expr]
        if show_filter:
            sql += " AND show IN (" + ",".join("?" * len(show_filter)) + ")"
            p.extend(show_filter)
        if lo:
            sql += " AND upload_date >= ?"
            p.append(lo)
        if hi:
            sql += " AND upload_date <= ?"
            p.append(hi)
        sql += " ORDER BY score LIMIT ?"
        p.append(limit)
        return db.execute(sql, p).fetchall()

    # With --rerank the candidate pool is the reranker's depth, since the
    # cross-encoder reorders everything it is handed. Without it, over-fetch
    # enough that the caps still have k passages to choose from.
    pool = args.rerank_depth if args.rerank else max(args.k * 8, 50)

    try:
        if args.max_per_show:
            # PER-SHOW QUERIES, THEN MERGE BY SCORE.
            #
            # A post-filter over one global result set CANNOT diversify -- it can
            # only drop results. With 5,585 DoThisNotThat chunks against 108 for
            # MarTech, the smaller show never reaches a global top-50 at all, so
            # capping the larger one just returned fewer passages (observed
            # 19 Aug: `--max-per-show 3 -k 6` gave 3, all from one show).
            # Querying each show separately gives every show a fair chance to
            # contribute its best passages; merging on bm25 keeps ranking honest.
            targets = shows or [r[0] for r in
                                db.execute("SELECT DISTINCT show FROM chunks").fetchall()]
            rows = []
            for s in targets:
                rows.extend(fetch([s], pool if args.rerank
                                  else max(args.max_per_show * 6, 20)))
            rows.sort(key=lambda r: r[6])       # bm25: lower is better
        else:
            rows = fetch(shows, pool)
    except sqlite3.OperationalError as exc:
        sys.exit(f"ABORT: query failed: {exc}")
    finally:
        db.close()

    # RERANK BEFORE CAPPING. The caps take from the top of the ordering, so
    # applying them to BM25 order and reranking the survivors would discard the
    # passages the cross-encoder would have promoted.
    if args.rerank and rows:
        rows = rerank(rows, query, args.rerank_model, args.fp16)

    picked, per_ep, per_show = [], {}, {}
    for r in rows:
        stem, show = r[2], r[7]
        if args.max_per_episode and per_ep.get(stem, 0) >= args.max_per_episode:
            continue
        if args.max_per_show and per_show.get(show, 0) >= args.max_per_show:
            continue
        per_ep[stem] = per_ep.get(stem, 0) + 1
        per_show[show] = per_show.get(show, 0) + 1
        picked.append(r)
        if len(picked) >= args.k:
            break

    if not picked:
        print(f"Nothing matched \"{query}\""
              + (f" in {', '.join(shows)}" if shows else "")
              + (" in that date range." if (lo or hi) else "."))
        print("Try fewer or more common words, widen the dates, "
              "or drop --show. `--shows` lists what is indexed.")
        return

    if args.format == "json":
        print(json.dumps([{
            "show": r[7], "header": r[1], "stem": r[2], "upload_date": r[3],
            "start_s": r[4], "end_s": r[5], "score": r[6],
            "text": r[0],
        } for r in picked], indent=2, ensure_ascii=False))
        return

    if args.format == "md":
        print(f"# Podcast context for: {query}\n")
        print(f"{len(picked)} passages retrieved by keyword relevance from "
              f"{len(per_show)} show(s). **Dates matter** -- these are tactical "
              f"marketing podcasts, and platform advice from an older episode "
              f"may no longer hold.\n")

    for i, (text, header, stem, date, start_s, end_s, score, show, _b) in enumerate(picked, 1):
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
        years = sorted({(r[3] or "")[:4] for r in picked if r[3]})
        counts = ", ".join(f"{s} {n}" for s, n in sorted(per_show.items()))
        print(f"---\n{len(picked)} passages, {len(per_ep)} episodes, "
              f"{'-'.join([years[0], years[-1]]) if years else 'undated'} "
              f"({counts}).  Query: {query}")


if __name__ == "__main__":
    main()
