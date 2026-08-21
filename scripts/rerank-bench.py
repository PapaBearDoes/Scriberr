#!/usr/bin/env python3
"""
rerank-bench.py -- what does reranking actually cost with the model resident?

THE CLAIM UNDER TEST. The CLI loads bge-reranker-v2-m3 on every invocation:
~2.3 GB, several seconds, then ~600 ms of scoring. A sidecar holding the model
resident should give ~600 ms flat. **That figure is measured time MINUS a cold
start -- an inference, not a measurement**, and it has been repeated enough
times to deserve checking before anything is built on it.

WHAT IT MEASURES, AND WHY EACH MATTERS TO THE SIDECAR DESIGN
  model load       the cost a sidecar pays ONCE and the CLI pays EVERY time.
                   The whole case for a resident service is this number.
  first query      CUDA autotunes kernels on the first inference, so query 1 is
                   usually slower than query 2. A service that idles then gets
                   one request pays this, not the steady-state number.
  steady state     p50/p90/p95/max. **The tail sets the failover timeout** -- a
                   threshold below p95 silently downgrades normal-but-slow
                   queries to BM25 order.
  after idle       with --idle, sleeps between queries. If the GPU context goes
                   cold between hourly requests, the resident advantage is
                   smaller than it looks and the sidecar may need a keep-warm.
  bm25 separately  an agent experiences fetch + rerank. Reporting only the
                   rerank half would understate what it waits for.

USAGE (needs the venv interpreter)
    ~/venvs/rerank/bin/python scripts/rerank-bench.py --n 40
    ~/venvs/rerank/bin/python scripts/rerank-bench.py --n 12 --idle 20
    ~/venvs/rerank/bin/python scripts/rerank-bench.py --n 40 --no-fp16
    ~/venvs/rerank/bin/python scripts/rerank-bench.py --n 40 --depth 100

STOP THE FEED TIMER FIRST, or you are measuring GPU contention with the
transcription backfill rather than the reranker.

    sudo systemctl stop scriberr-feed.timer     # and start it again after
"""

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time

DEFAULT_INDEX = "/storage/nas/ai/scriberr/index/chunks.sqlite"
DEFAULT_ANALYSIS = "/storage/nas/ai/scriberr/analysis"
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def fts_expr(text, min_len=2):
    """Same construction as search-corpus.py: OR'd terms, minimum length 2."""
    toks = [t for t in WORD_RE.findall((text or "").lower()) if len(t) >= min_len]
    if not toks:
        return None
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks[:60])


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarise(name, values):
    if not values:
        print(f"  {name:<16} no samples")
        return
    print(f"  {name:<16} p50 {statistics.median(values):>7.0f}   "
          f"p90 {pct(values, 90):>7.0f}   p95 {pct(values, 95):>7.0f}   "
          f"max {max(values):>7.0f}   mean {statistics.fmean(values):>7.0f}  (ms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=40, help="queries to time")
    ap.add_argument("--depth", type=int, default=50, help="candidates per query")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="seconds to sleep between queries, to test whether the "
                         "GPU context goes cold between requests")
    args = ap.parse_args()

    if not os.path.exists(args.index):
        sys.exit(f"ABORT: no index at {args.index}")
    eval_path = os.path.join(args.analysis, "eval-candidates.jsonl")
    if not os.path.exists(eval_path):
        sys.exit(f"ABORT: no eval set at {eval_path} -- run measure-corpus.py first")

    # Real queries, not synthetic ones -- term count drives both the FTS scan and
    # the tokenised sequence length, so made-up queries would misreport both.
    queries = []
    with open(eval_path, encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line).get("publisher_description")
            if q and len(WORD_RE.findall(q)) >= 5:
                queries.append(q)
    if not queries:
        sys.exit("ABORT: no usable queries in the eval set")
    step = max(1, len(queries) // args.n)
    queries = [queries[i * step] for i in range(min(args.n, len(queries) // step))]

    db = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    n_chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        sys.exit("ABORT: sentence-transformers not installed.\n"
                 "       run this with ~/venvs/rerank/bin/python")

    device, kwargs = "cpu", {}
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            if args.fp16:
                kwargs = {"model_kwargs": {"torch_dtype": torch.float16}}
    except ImportError:
        pass

    # THE NUMBER THE WHOLE SIDECAR CASE RESTS ON: what the CLI pays every
    # invocation and a resident service pays once.
    t0 = time.perf_counter()
    try:
        model = CrossEncoder(args.model, device=device, max_length=512, **kwargs)
    except Exception as exc:
        print(f"  fp16 unavailable ({type(exc).__name__}), falling back to fp32",
              file=sys.stderr)
        model = CrossEncoder(args.model, device=device, max_length=512)
    load_ms = (time.perf_counter() - t0) * 1000

    bm25_ms, rerank_ms, total_ms, cand_counts = [], [], [], []
    for i, q in enumerate(queries):
        if args.idle and i:
            time.sleep(args.idle)
        expr = fts_expr(q)
        if not expr:
            continue

        t = time.perf_counter()
        rows = db.execute(
            "SELECT body FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
            (expr, args.depth)).fetchall()
        b_ms = (time.perf_counter() - t) * 1000
        if not rows:
            continue

        t = time.perf_counter()
        scores = model.predict([(q, r[0]) for r in rows],
                               batch_size=args.batch_size, show_progress_bar=False)
        sorted(range(len(rows)), key=lambda j: -scores[j])   # the real sort, timed
        r_ms = (time.perf_counter() - t) * 1000

        bm25_ms.append(b_ms)
        rerank_ms.append(r_ms)
        total_ms.append(b_ms + r_ms)
        cand_counts.append(len(rows))

    db.close()

    if not rerank_ms:
        sys.exit("ABORT: no queries produced candidates")

    fp = "fp16" if (device == "cuda" and args.fp16) else "fp32"
    print()
    print(f"{args.model}  {device}/{fp}  depth {args.depth}  batch {args.batch_size}")
    print(f"index {n_chunks:,} chunks   {len(rerank_ms)} queries   "
          f"median {int(statistics.median(cand_counts))} candidates each"
          + (f"   idle {args.idle}s between queries" if args.idle else ""))
    print()
    print(f"  model load       {load_ms:>7.0f} ms"
          f"   <- paid EVERY invocation by the CLI, ONCE by a sidecar")
    print(f"  first query      {rerank_ms[0]:>7.0f} ms"
          f"   <- CUDA autotunes on first inference")
    if len(rerank_ms) > 1:
        rest = rerank_ms[1:]
        print(f"  steady state     {statistics.median(rest):>7.0f} ms (p50 of the rest)")
    print()
    summarise("bm25 fetch", bm25_ms)
    summarise("rerank", rerank_ms)
    summarise("total", total_ms)

    print()
    cli_cold = load_ms + statistics.median(rerank_ms)
    resident = statistics.median(total_ms)
    print(f"  CLI today, per query      {cli_cold:>8.0f} ms  (load + rerank)")
    print(f"  resident sidecar          {resident:>8.0f} ms  (fetch + rerank)")
    print(f"  saved per query           {cli_cold - resident:>8.0f} ms")
    print()
    p95 = pct(total_ms, 95)
    print(f"  SUGGESTED FAILOVER TIMEOUT  {max(1000, p95 * 2):>6.0f} ms  "
          f"(2x p95 of {p95:.0f})")
    print(f"  A threshold below p95 silently downgrades normal-but-slow queries")
    print(f"  to BM25 order. Too high and a busy GPU stalls the agent instead.")
    if args.idle:
        print()
        print(f"  Ran with {args.idle}s idle between queries. Compare the steady")
        print(f"  state against a run WITHOUT --idle: if it is materially slower")
        print(f"  here, the GPU context goes cold between requests and the sidecar")
        print(f"  needs a keep-warm ping rather than just staying resident.")


if __name__ == "__main__":
    main()
