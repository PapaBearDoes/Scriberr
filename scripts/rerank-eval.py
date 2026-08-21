#!/usr/bin/env python3
"""
rerank-eval.py -- does a cross-encoder reranker actually beat BM25 alone?

PROVE IT BEFORE WIRING IT. This reads the EXISTING index and eval set, takes
BM25's top-N candidates, reorders them with a cross-encoder, and prints the
same recall table before and after. Nothing is wired into search-corpus.py
until this shows a real gain -- an afternoon spent here is cheaper than a
heavyweight dependency baked into the tool you actually use.

WHY A RERANKER AND NOT EMBEDDINGS. Measured across three index sizes: as the
corpus grew from 5,693 to 20,226 chunks, misses with 60%+ query-term overlap
rose 53.2% -> 60.7% while misses with ZERO vocabulary overlap fell 3.0% -> 1.9%.
Scale made RANKING worse and vocabulary coverage better. Ranking is what a
reranker fixes; vocabulary mismatch is the only thing dense retrieval uniquely
fixes, and it is under 2% of misses. R@5 0.757 against R@50 0.885 says BM25
already retrieves the right chunk and orders it badly.

THE CEILING IS R@DEPTH, AND IT IS REPORTED. A reranker can only reorder what
BM25 handed it. If recall@50 is 0.885, no amount of reranking gets recall@5
above 0.885 -- so the honest question is not "did it improve" but "how much of
the available headroom did it capture". Both are printed.

USAGE
    python3 scripts/rerank-eval.py --sample 500          # quick, ~25k pairs
    python3 scripts/rerank-eval.py --sample 1500         # the real run
    python3 scripts/rerank-eval.py --model BAAI/bge-reranker-v2-m3
    python3 scripts/rerank-eval.py --device cpu

REQUIRES torch + sentence-transformers (~2.5 GB):
    pip install --user sentence-transformers

GPU CONTENTION: the transcription backfill uses the same GPU. Stop the feed
timer for the duration of a real run, or expect this to take considerably
longer than the idle-card estimate.

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
from collections import Counter, defaultdict

DEFAULT_INDEX = "/storage/nas/ai/scriberr/index/chunks.sqlite"
DEFAULT_ANALYSIS = "/storage/nas/ai/scriberr/analysis"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Must match build-index.py. A tier-1 hit needs the right episode AND a chunk
# overlapping the moment timestamp with this slack.
HIT_BEFORE = 30.0
HIT_AFTER = 90.0


def fts_expr(text, min_len=2):
    """Same query construction as build-index.py: OR'd terms, min length 2.
    Length 3 silently discards "ai"; length 4 also loses "seo", "cta", "roi"."""
    toks = [t for t in WORD_RE.findall((text or "").lower()) if len(t) >= min_len]
    if not toks:
        return None
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks[:60])


def is_hit(row, cand):
    """cand = (show, stem, start_s, end_s, body)."""
    show, stem, start_s, end_s, _body = cand
    if show != row["show"] or stem != row["stem"]:
        return False
    if row["tier"] in ("episode", "title"):
        return True                              # episode-level target
    t = row["t_seconds"]
    return end_s >= t - HIT_BEFORE and start_s <= t + HIT_AFTER


def rank_of(row, cands):
    for i, c in enumerate(cands, 1):
        if is_hit(row, c):
            return i
    return None


def load_model(name, device, max_length=512, fp16=False):
    # Before touching the Hub: sentence-transformers checks for model updates on
    # every load, and unauthenticated requests get the slow tier plus a warning.
    import _hf_env
    _hf_env.load()
    print(f"  {_hf_env.describe()}", file=sys.stderr)
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        sys.exit("ABORT: sentence-transformers is not installed.\n"
                 "       python3 -m venv ~/venvs/rerank\n"
                 "       ~/venvs/rerank/bin/pip install sentence-transformers")
    print(f"loading {name} on {device}"
          + (" (fp16)" if fp16 else "") + " ...", file=sys.stderr)
    if fp16 and device != "cpu":
        # Roughly halves latency and memory on a large reranker. Not every
        # sentence-transformers version accepts model_kwargs, so fall back
        # loudly rather than failing the run.
        try:
            import torch
            return CrossEncoder(name, device=device, max_length=max_length,
                                model_kwargs={"torch_dtype": torch.float16})
        except Exception as exc:
            print(f"  fp16 unavailable ({type(exc).__name__}), using fp32",
                  file=sys.stderr)
    return CrossEncoder(name, device=device, max_length=max_length)


def pick_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def summarise(ranks_by_tier, ranks_by_show, ks, depth):
    """ranks_* map key -> list of rank-or-None."""
    def recall(ranks, k):
        return round(sum(1 for r in ranks if r and r <= k) / len(ranks), 4) if ranks else 0

    allr = [r for v in ranks_by_tier.values() for r in v]
    out = {
        "n": len(allr),
        "recall": {k: recall(allr, k) for k in ks},
        "ceiling_recall_at_depth": recall(allr, depth),
        "mrr": round(statistics.fmean([1.0 / r if r else 0.0 for r in allr]), 4) if allr else 0,
        "by_tier": {t: {"n": len(v), **{f"recall@{k}": recall(v, k) for k in ks}}
                    for t, v in sorted(ranks_by_tier.items())},
        "by_show": {s: {"n": len(v), **{f"recall@{k}": recall(v, k) for k in ks}}
                    for s, v in sorted(ranks_by_show.items())},
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--depth", type=int, default=50,
                    help="BM25 candidates to rerank. R@50 is 0.885 and R@100 only "
                         "0.905 -- the second fifty doubles cost for 2 points of ceiling.")
    ap.add_argument("--sample", type=int, default=500,
                    help="eval queries; 0 = all")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="drop to 16 for a large reranker or you will OOM a 3080")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--fp16", action="store_true",
                    help="half precision; roughly halves latency on a large model")
    ap.add_argument("--show", default=None, help="comma-separated show names")
    ap.add_argument("--tier", default=None, help="comma-separated: chunk,episode,title")
    ap.add_argument("--dump-misses", nargs="?", const="AUTO", default=None,
                    help="write surviving misses plus the target-score diagnostic")
    args = ap.parse_args()

    if not os.path.exists(args.index):
        sys.exit(f"ABORT: no index at {args.index} -- run build-index.py first")
    eval_path = os.path.join(args.analysis, "eval-candidates.jsonl")
    if not os.path.exists(eval_path):
        sys.exit(f"ABORT: no eval set at {eval_path} -- run measure-corpus.py first")

    only_shows = {s.strip() for s in args.show.split(",")} if args.show else None
    only_tiers = {t.strip() for t in args.tier.split(",")} if args.tier else None

    rows = []
    with open(eval_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if only_shows and r["show"] not in only_shows:
                continue
            if only_tiers and r["tier"] not in only_tiers:
                continue
            rows.append(r)
    if not rows:
        sys.exit("ABORT: no eval rows matched those filters")
    if args.sample and args.sample < len(rows):
        step = len(rows) / args.sample          # deterministic, spread across shows
        rows = [rows[int(i * step)] for i in range(args.sample)]

    db = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)

    # CONTAMINATION GUARD. The reranker scores (query, passage) pairs against
    # `body`. If the index was built with --index-header, `body` begins with the
    # header -- which contains the episode TITLE, and tier-3 queries ARE the
    # title. That would hand the reranker the answer key and every tier-3 number
    # would be meaningless.
    probe = db.execute("SELECT body, header FROM chunks LIMIT 1").fetchone()
    if probe and probe[0].startswith(probe[1]):
        sys.exit("ABORT: this index was built with --index-header, so chunk bodies\n"
                 "       contain the episode title. Tier-3 queries ARE the title,\n"
                 "       so reranking against it would score the answer key.\n"
                 "       Rebuild without --index-header.")

    device = pick_device(args.device)
    model = load_model(args.model, device, args.max_length, args.fp16)

    ks = [1, 3, 5, 10]
    base_tier, base_show = defaultdict(list), defaultdict(list)
    rr_tier, rr_show = defaultdict(list), defaultdict(list)
    collect = args.dump_misses is not None
    records = []
    empty = 0
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        expr = fts_expr(row["publisher_description"])
        if not expr:
            empty += 1
            continue
        cands = db.execute(
            "SELECT show, stem, start_s, end_s, body FROM chunks"
            " WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
            (expr, args.depth)).fetchall()
        if not cands:
            empty += 1
            continue

        b = rank_of(row, cands)
        base_tier[row["tier"]].append(b)
        base_show[row["show"]].append(b)

        scores = model.predict([(row["publisher_description"], c[4]) for c in cands],
                               batch_size=args.batch_size, show_progress_bar=False)
        order = sorted(range(len(cands)), key=lambda j: -scores[j])
        reranked = [cands[j] for j in order]

        r = rank_of(row, reranked)
        rr_tier[row["tier"]].append(r)
        rr_show[row["show"]].append(r)

        if collect:
            # THE UNBIASED INSTRUMENT. A cross-encoder emits an ABSOLUTE
            # relevance score for any (query, passage) pair, so we can score the
            # LABELLED TARGET directly -- and the target was not chosen by the
            # model, so this comparison carries none of the selection bias that
            # made the term-overlap false-miss proxy worthless (it returned 86%
            # flat, 83% IDF-weighted, and both were rigged).
            #
            # The question it answers: when reranking fails, is the target
            # scored HIGH and merely outranked -- a genuine ranking failure the
            # model could in principle fix -- or scored LOW, meaning the
            # labelled passage does not actually answer the query and the LABEL
            # is what is wrong? Compare the target-score distribution for hits
            # against misses; if they overlap, the labels are fine and the model
            # is the limit. If missed targets score far lower, no reranker of
            # any size will help because those queries are not broken retrieval.
            hit_idx = [j for j, c in enumerate(cands) if is_hit(row, c)]
            tgt = max((float(scores[j]) for j in hit_idx), default=None)
            top_j = order[0]
            records.append({
                "tier": row["tier"], "show": row["show"], "stem": row["stem"],
                "query": row["publisher_description"],
                "target_in_candidates": bool(hit_idx),
                "bm25_rank": b, "rerank_rank": r,
                "hit_at_5": bool(r and r <= 5),
                "target_score": round(tgt, 4) if tgt is not None else None,
                "top_score": round(float(scores[top_j]), 4),
                "top_show": cands[top_j][0], "top_stem": cands[top_j][1],
                "top_excerpt": cands[top_j][4][:400],
            })

        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  {el:.0f}s elapsed, "
                  f"{el / i * (len(rows) - i):.0f}s left", file=sys.stderr)

    db.close()
    elapsed = time.time() - t0

    base = summarise(base_tier, base_show, ks, args.depth)
    rerank = summarise(rr_tier, rr_show, ks, args.depth)

    result = {
        "model": args.model, "device": device, "depth": args.depth,
        "n_queries": base["n"], "queries_with_no_candidates": empty,
        "seconds": round(elapsed, 1),
        "baseline": base, "reranked": rerank,
    }
    os.makedirs(args.analysis, exist_ok=True)
    # Per-model filename so comparing two rerankers does not overwrite the first.
    # NOTE: results are only comparable if eval-candidates.jsonl and the index
    # are unchanged between runs -- do NOT re-run measure-corpus.py or
    # build-index.py in the middle of a model comparison.
    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-").lower()
    out_path = os.path.join(args.analysis, f"rerank-eval-{slug}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- report
    print()
    print(f"{args.model}  on {device}, depth {args.depth}, {base['n']} queries, "
          f"{elapsed:.0f}s ({elapsed / max(base['n'], 1) * 1000:.0f} ms/query)")
    print()
    print(f"{'':<12} " + " ".join(f"{'R@'+str(k):>8}" for k in ks) + f" {'MRR':>8}")
    print(f"{'BM25':<12} " + " ".join(f"{base['recall'][k]:>8.4f}" for k in ks)
          + f" {base['mrr']:>8.4f}")
    print(f"{'reranked':<12} " + " ".join(f"{rerank['recall'][k]:>8.4f}" for k in ks)
          + f" {rerank['mrr']:>8.4f}")
    print(f"{'delta':<12} " + " ".join(
        f"{rerank['recall'][k] - base['recall'][k]:>+8.4f}" for k in ks)
        + f" {rerank['mrr'] - base['mrr']:>+8.4f}")
    print()

    ceil_ = base["ceiling_recall_at_depth"]
    gap = ceil_ - base["recall"][5]
    got = rerank["recall"][5] - base["recall"][5]
    print(f"CEILING: reranking can only reorder what BM25 retrieved, so "
          f"recall@{args.depth} = {ceil_:.4f} is the hard limit.")
    print(f"         headroom at k=5 was {gap:+.4f}; reranking captured "
          f"{got:+.4f}"
          + (f" ({got / gap * 100:.0f}% of it)" if gap > 0 else ""))
    print()
    print("per tier (chunk is the tightest target; title the loosest)")
    for t in sorted(rerank["by_tier"]):
        b5 = base["by_tier"][t]["recall@5"]
        r5 = rerank["by_tier"][t]["recall@5"]
        print(f"  {t:<10} n={base['by_tier'][t]['n']:<5} "
              f"BM25 {b5:.4f} -> {r5:.4f}  ({r5 - b5:+.4f})")
    print()
    print("per show")
    for s in sorted(rerank["by_show"]):
        b5 = base["by_show"][s]["recall@5"]
        r5 = rerank["by_show"][s]["recall@5"]
        print(f"  {s:<18} n={base['by_show'][s]['n']:<5} "
              f"BM25 {b5:.4f} -> {r5:.4f}  ({r5 - b5:+.4f})")
    if empty:
        print()
        print(f"NOTE: {empty} queries returned no BM25 candidates and were excluded "
              f"from BOTH figures.")

    if collect:
        hits = [x for x in records if x["hit_at_5"] and x["target_score"] is not None]
        miss_in = [x for x in records if not x["hit_at_5"] and x["target_in_candidates"]]
        miss_out = [x for x in records if not x["hit_at_5"] and not x["target_in_candidates"]]

        path = (args.dump_misses if args.dump_misses != "AUTO"
                else os.path.join(args.analysis, f"rerank-misses-{slug}.jsonl"))
        with open(path, "w", encoding="utf-8") as fh:
            for x in miss_in + miss_out:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")

        def med(v):
            return round(statistics.median(v), 4) if v else None

        hit_med = med([x["target_score"] for x in hits])
        miss_scores = [x["target_score"] for x in miss_in]
        strong = sum(1 for s in miss_scores if hit_med is not None and s >= hit_med)
        weak = len(miss_scores) - strong
        n_miss = len(miss_in) + len(miss_out)

        print()
        print(f"MISS ANATOMY  ({n_miss} misses at k=5)")
        print(f"  target never retrieved by BM25       {len(miss_out):>4}"
              f"   <- retrieval failure; only more depth or a different retriever helps")
        print(f"  target retrieved but ranked >5       {len(miss_in):>4}"
              f"   <- ranking failure OR label noise; the split is below")
        print()
        print(f"  median target relevance score, when the query HIT     {hit_med}")
        print(f"  median target relevance score, when it MISSED         {med(miss_scores)}")
        print(f"  median score of what the model ranked FIRST instead   "
              f"{med([x['top_score'] for x in miss_in])}")
        print()
        if miss_in:
            print(f"  missed targets scoring >= the hit median  {strong:>4} "
                  f"({100.0 * strong / len(miss_in):.0f}%)"
                  f"   <- model rated the target well and still ranked it low:")
            print(f"                                                    "
                  f"       GENUINE ranking failure, a better model may fix it")
            print(f"  missed targets scoring below it           {weak:>4} "
                  f"({100.0 * weak / len(miss_in):.0f}%)"
                  f"   <- the labelled passage scores poorly on its own query:")
            print(f"                                                    "
                  f"       the LABEL is suspect, no reranker can fix these")
        print()
        print("  This is the unbiased version of the term-overlap proxy that returned")
        print("  86% and had to be thrown out: the target is scored by the model but")
        print("  was never SELECTED by it, so there is no selection bias here.")
        print(f"  wrote {path}")

    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
