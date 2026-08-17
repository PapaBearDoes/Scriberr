#!/usr/bin/env python3
"""
build-index.py -- chunk the corpus, index it with SQLite FTS5, and score the
result against the labelled eval set. One pass, one number.

WHY FTS5 FIRST. The corpus measured at 2.03M tokens across 121,401 segments,
which is small. Before committing to an embedding model, a vector store, and a
service to keep running, there should be a number that a vector store has to
beat. This produces that number. If BM25 gets recall@5 of 0.8 on 3,915 labelled
queries, the bar for dense retrieval is "beat 0.8 by enough to justify a second
system" rather than "feels better".

THE EVAL SET IS NOT A TRAINING SET -- READ THIS BEFORE CHANGING THE CHUNKER.
Tier 1 queries ARE the publisher's Best Moments descriptions, and those same
timestamps are what the `moments` chunker splits on. Splitting on them is fine:
a boundary carries no text. Writing the descriptions INTO the chunk body would
not be fine -- it would index the answer key and every score would be garbage.
Nothing in this file puts a moment description into a chunk. Keep it that way.

For the same reason the episode title and date go in an UNINDEXED column by
default. Publisher descriptions echo the title constantly, so indexing the title
inflates recall without improving real retrieval. `--index-header` exists to
measure that effect, not to be left on.

USAGE
    python3 scripts/build-index.py                    # build + score, defaults
    python3 scripts/build-index.py --sweep            # parameter grid
    python3 scripts/build-index.py --eval-sample 800  # faster iteration
    python3 scripts/build-index.py --query "subject line tips"

THE EXPERIMENT THAT MATTERS is `--sweep`, because it includes `window` as a
chunk mode. If moments-based chunking does not beat a naive fixed window, the
publisher boundaries are not worth the complexity and we use the window.

OUTPUTS
    <out>/index-eval.json     every configuration scored, machine-readable
    <index>                   the built index, default chunks.sqlite on the NAS

SQLITE ON NFS: the index is built in a local temp file and copied to its final
location. SQLite locking over NFS is a known hazard and a half-written index
that reports success is exactly the failure mode this project keeps hitting.
"""

import argparse
import html
import json
import os
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

DEFAULT_TRANSCRIPTS = "/storage/nas/ai/scriberr/transcripts"
DEFAULT_AUDIO = "/storage/nas/ai/scriberr/audio"
DEFAULT_ANALYSIS = "/storage/nas/ai/scriberr/analysis"
DEFAULT_INDEX = "/storage/nas/ai/scriberr/index/chunks.sqlite"

TAG_RE = re.compile(r"<[^>]+>")
PARA_RE = re.compile(r"</p\s*>|<br\s*/?>", re.I)
MOMENT_RE = re.compile(r"^\(?(\d{1,2}:\d{2}(?::\d{2})?)\)?\s*[-\u2013]?\s*(.+)$")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
NORM_RE = re.compile(r"[^a-z0-9 ]+")

# A tier-1 hit counts if the chunk overlaps the moment timestamp with this much
# slack. Publisher timestamps are hand-typed and land a little before the
# passage they describe more often than after it, hence the asymmetry.
HIT_BEFORE = 30.0
HIT_AFTER = 90.0


# --------------------------------------------------------------------- shared

def strip_html(raw):
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
    out = []
    for line in strip_html(description).split("\n"):
        m = MOMENT_RE.match(line.strip())
        if not m or not m.group(2).strip():
            continue
        try:
            out.append((to_seconds(m.group(1)), m.group(2).strip()))
        except ValueError:
            pass
    out.sort(key=lambda x: x[0])
    return out


def words(text):
    return len(WORD_RE.findall(text or ""))


def normalize(text):
    return NORM_RE.sub("", (text or "").lower()).strip()


# ---------------------------------------------------------------- corpus load

def load_corpus(transcripts_dir, audio_dir, limit=0):
    """Read every episode once. ~10 MB of text; held in memory so a sweep does
    not re-read 1,270 files per configuration."""
    stems = sorted(f[:-5] for f in os.listdir(transcripts_dir) if f.endswith(".json"))
    if limit:
        stems = stems[:limit]
    episodes = []
    for stem in stems:
        try:
            with open(os.path.join(transcripts_dir, stem + ".json"), encoding="utf-8", errors="replace") as fh:
                tj = json.load(fh)
        except Exception as exc:
            print(f"  skipping {stem}: {type(exc).__name__}", file=sys.stderr)
            continue
        side = {}
        spath = os.path.join(audio_dir, stem + ".info.json")
        if os.path.exists(spath):
            try:
                with open(spath, encoding="utf-8", errors="replace") as fh:
                    side = json.load(fh)
            except Exception:
                pass
        segs = (tj.get("transcript") or {}).get("segments") or []
        episodes.append({
            "stem": stem,
            "title": tj.get("title") or side.get("title") or stem,
            "episode_number": side.get("episode_number"),
            "upload_date": side.get("upload_date"),
            "segments": segs,
            "moments": parse_moments(side.get("description", "")),
        })
    return episodes


# ----------------------------------------------------------------- boilerplate

def build_stoplist(episodes, min_episodes):
    """
    Derive the boilerplate stoplist FROM THE CORPUS, never from a hand-written
    list. Measured 17 Aug 2026: the intro exists as at least three variants, and
    `welcome to do this not that the podcast for marketers` (278 episodes) and
    `...from marketers` (177) are the same spoken line landing two ways. A
    handwritten list catches one and silently misses the other.

    Frequency across DISTINCT EPISODES is the whole rule. Position is not used:
    it would miss the mid-roll cross-promo that sits around 75% through the
    ~112 Bathroom Break episodes. A sentence appearing verbatim in 20+ separate
    episodes is boilerplate by definition, whatever position it occupies.
    """
    if not min_episodes:
        return set()
    seen = Counter()
    for ep in episodes:
        for norm in {normalize(s.get("text")) for s in ep["segments"]}:
            if len(norm) >= 15:
                seen[norm] += 1
    return {t for t, c in seen.items() if c >= min_episodes}


# -------------------------------------------------------------------- chunking

def split_oversized(group, max_words):
    """Break a too-large chunk at segment boundaries. Parakeet emits clean
    sentences, so a break here never lands mid-sentence."""
    if sum(words(s.get("text")) for s in group) <= max_words:
        return [group]
    out, cur, n = [], [], 0
    for seg in group:
        cur.append(seg)
        n += words(seg.get("text"))
        if n >= max_words:
            out.append(cur)
            cur, n = [], 0
    if cur:
        if out and n < max_words * 0.4:      # don't leave a stub
            out[-1].extend(cur)
        else:
            out.append(cur)
    return out


def chunk_window(segments, target_words, overlap):
    out, cur, n = [], [], 0
    for seg in segments:
        cur.append(seg)
        n += words(seg.get("text"))
        if n >= target_words:
            out.append(cur)
            cur = cur[-overlap:] if overlap else []
            n = sum(words(s.get("text")) for s in cur)
    if cur and len(cur) > overlap:
        out.append(cur)
    return out


def chunk_moments(segments, moments, max_words, target_words, overlap):
    if not moments:
        return chunk_window(segments, target_words, overlap)
    bounds = [m[0] for m in moments]
    groups, cur, idx = [], [], 0
    for seg in segments:
        while idx < len(bounds) and float(seg.get("start", 0)) >= bounds[idx]:
            if cur:
                groups.append(cur)
            cur, idx = [], idx + 1
        cur.append(seg)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        out.extend(split_oversized(g, max_words))
    return out


def chunk_episode(ep, cfg, stoplist):
    segs = [s for s in ep["segments"] if normalize(s.get("text")) not in stoplist]
    if not segs:
        return []
    if cfg["mode"] == "window":
        groups = chunk_window(segs, cfg["target_words"], cfg["overlap"])
    else:
        groups = chunk_moments(segs, ep["moments"], cfg["max_words"],
                               cfg["target_words"], cfg["overlap"])

    date = ep["upload_date"] or ""
    pretty = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date
    # The header exists so the CONSUMING model sees when this was said. Undated
    # 2023 platform tactics read as current advice, and the show is tactical.
    header = f"{ep['title']} | episode {ep['episode_number']} | {pretty}"

    rows = []
    for g in groups:
        body = " ".join((s.get("text") or "").strip() for s in g).strip()
        if not body:
            continue
        rows.append({
            "stem": ep["stem"],
            "episode_number": ep["episode_number"],
            "upload_date": date,
            "start_s": float(g[0].get("start", 0)),
            "end_s": float(g[-1].get("end", 0)),
            "speakers": ",".join(sorted({s.get("speaker") or "" for s in g})),
            "header": header,
            "body": (header + "\n" + body) if cfg["index_header"] else body,
            "text_for_model": header + "\n" + body,
        })
    return rows


# ----------------------------------------------------------------------- index

def build_index(path, rows):
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
        db.execute("DROP TABLE probe")
    except sqlite3.OperationalError:
        sys.exit("ABORT: this python3 sqlite3 has no FTS5. Check the build.")
    db.execute("""
        CREATE VIRTUAL TABLE chunks USING fts5(
            body,
            text_for_model UNINDEXED,
            header UNINDEXED,
            stem UNINDEXED,
            episode_number UNINDEXED,
            upload_date UNINDEXED,
            start_s UNINDEXED,
            end_s UNINDEXED,
            speakers UNINDEXED,
            tokenize='porter unicode61'
        )
    """)
    db.executemany(
        "INSERT INTO chunks (body, text_for_model, header, stem, episode_number,"
        " upload_date, start_s, end_s, speakers)"
        " VALUES (:body,:text_for_model,:header,:stem,:episode_number,"
        " :upload_date,:start_s,:end_s,:speakers)",
        rows,
    )
    db.commit()
    return db


def fts_query(text):
    """
    Free text -> a safe FTS5 MATCH expression.

    Bare multi-word MATCH is implicit AND, which returns nothing for a 20-word
    publisher description. OR plus bm25 ranking is the right shape: every term
    contributes, rare terms dominate the score. Each token is quoted so
    apostrophes and FTS operators cannot break the parse.
    """
    toks = [t for t in WORD_RE.findall(text.lower()) if len(t) > 2]
    if not toks:
        return None
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks[:60])


def search(db, query, k):
    expr = fts_query(query)
    if not expr:
        return []
    cur = db.execute(
        "SELECT stem, start_s, end_s, bm25(chunks) AS score FROM chunks"
        " WHERE chunks MATCH ? ORDER BY score LIMIT ?", (expr, k))
    return cur.fetchall()


# ------------------------------------------------------------------ evaluation

def evaluate(db, eval_rows, ks):
    maxk = max(ks)
    hits = {k: Counter() for k in ks}
    totals = Counter()
    rr = defaultdict(list)
    by_year = defaultdict(lambda: {"n": 0, "hit": 0})

    for row in eval_rows:
        tier = row["tier"]
        totals[tier] += 1
        year = (row.get("upload_date") or "????")[:4]
        by_year[year]["n"] += 1

        results = search(db, row["publisher_description"], maxk)
        rank = None
        for i, (stem, start_s, end_s, _score) in enumerate(results, 1):
            if stem != row["stem"]:
                continue
            if tier == "episode":
                rank = i
                break
            t = row["t_seconds"]
            if end_s >= t - HIT_BEFORE and start_s <= t + HIT_AFTER:
                rank = i
                break

        rr[tier].append(1.0 / rank if rank else 0.0)
        if rank:
            for k in ks:
                if rank <= k:
                    hits[k][tier] += 1
            if rank <= 5:
                by_year[year]["hit"] += 1

    out = {"totals": dict(totals), "recall": {}, "mrr": {}, "by_year": {}}
    for k in ks:
        out["recall"][k] = {
            t: round(hits[k][t] / totals[t], 4) for t in totals if totals[t]
        }
        all_n = sum(totals.values())
        out["recall"][k]["all"] = round(sum(hits[k].values()) / all_n, 4) if all_n else 0
    for t, vals in rr.items():
        out["mrr"][t] = round(statistics.fmean(vals), 4) if vals else 0
    for y, d in sorted(by_year.items()):
        out["by_year"][y] = {"n": d["n"], "recall@5": round(d["hit"] / d["n"], 4) if d["n"] else 0}
    return out


# ---------------------------------------------------------------------- driver

def run_config(episodes, eval_rows, cfg, ks, index_path=None):
    stoplist = build_stoplist(episodes, cfg["min_episodes"])
    rows = []
    for ep in episodes:
        rows.extend(chunk_episode(ep, cfg, stoplist))

    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    db = build_index(tmp.name, rows)
    try:
        scores = evaluate(db, eval_rows, ks)
    finally:
        db.close()

    wc = [words(r["body"]) for r in rows]
    result = {
        "config": cfg,
        "stoplist_size": len(stoplist),
        "n_chunks": len(rows),
        "chunk_words": {
            "median": round(statistics.median(wc), 1) if wc else 0,
            "mean": round(statistics.fmean(wc), 1) if wc else 0,
            "max": max(wc) if wc else 0,
        },
        "scores": scores,
    }

    if index_path:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        shutil.copy2(tmp.name, index_path)     # build local, copy to NFS
        result["index_path"] = index_path
    os.unlink(tmp.name)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--audio", default=DEFAULT_AUDIO)
    ap.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--limit", type=int, default=0, help="episodes to load (smoke test)")
    ap.add_argument("--eval-sample", type=int, default=0, help="score against N sampled queries")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--query", help="search the built index and print results, no scoring")
    ap.add_argument("--mode", default="moments", choices=["moments", "window"])
    ap.add_argument("--target-words", type=int, default=250)
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--min-episodes", type=int, default=20,
                    help="boilerplate threshold; 0 disables stripping")
    ap.add_argument("--index-header", action="store_true",
                    help="also index title and date (inflates recall, see docstring)")
    args = ap.parse_args()

    print("loading corpus ...", file=sys.stderr)
    episodes = load_corpus(args.transcripts, args.audio, args.limit)
    if not episodes:
        sys.exit("ABORT: no episodes loaded")
    print(f"  {len(episodes)} episodes", file=sys.stderr)

    eval_path = os.path.join(args.analysis, "eval-candidates.jsonl")
    if not os.path.exists(eval_path):
        sys.exit(f"ABORT: no eval set at {eval_path} -- run measure-corpus.py first")
    stems = {e["stem"] for e in episodes}
    eval_rows = []
    with open(eval_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["stem"] in stems:
                eval_rows.append(r)
    if args.eval_sample and args.eval_sample < len(eval_rows):
        step = len(eval_rows) / args.eval_sample      # deterministic, spread across years
        eval_rows = [eval_rows[int(i * step)] for i in range(args.eval_sample)]
    print(f"  {len(eval_rows)} eval queries", file=sys.stderr)

    base = {
        "mode": args.mode, "target_words": args.target_words,
        "max_words": args.max_words, "overlap": args.overlap,
        "min_episodes": args.min_episodes, "index_header": args.index_header,
    }
    ks = [1, 3, 5, 10]

    if args.query:
        res = run_config(episodes, [], base, ks, index_path=args.index)
        db = sqlite3.connect(args.index)
        print()
        for stem, start_s, end_s, score in search(db, args.query, 5):
            print(f"  {score:8.2f}  {int(start_s):5d}s  {stem[:80]}")
        db.close()
        return

    configs = [base]
    if args.sweep:
        configs = []
        for mode in ("moments", "window"):
            for tw in (200, 250, 350):
                for me in (0, 20):
                    configs.append({**base, "mode": mode, "target_words": tw,
                                    "min_episodes": me})

    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg['mode']} tw={cfg['target_words']} "
              f"boilerplate>={cfg['min_episodes']} ...", file=sys.stderr)
        keep = args.index if (len(configs) == 1) else None
        results.append(run_config(episodes, eval_rows, cfg, ks, index_path=keep))

    os.makedirs(args.analysis, exist_ok=True)
    out_path = os.path.join(args.analysis, "index-eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print()
    print(f"{'mode':<8} {'tw':>4} {'bp':>3} {'chunks':>7} {'med_w':>6} "
          f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR-c':>7} {'MRR-e':>7}")
    for r in results:
        c, s = r["config"], r["scores"]
        print(f"{c['mode']:<8} {c['target_words']:>4} {c['min_episodes']:>3} "
              f"{r['n_chunks']:>7} {r['chunk_words']['median']:>6} "
              f"{s['recall'][1].get('all', 0):>7.4f} "
              f"{s['recall'][5].get('all', 0):>7.4f} "
              f"{s['recall'][10].get('all', 0):>7.4f} "
              f"{s['mrr'].get('chunk', 0):>7.4f} "
              f"{s['mrr'].get('episode', 0):>7.4f}")

    best = max(results, key=lambda r: r["scores"]["recall"][5].get("all", 0))
    print()
    print("best by recall@5:", json.dumps(best["config"]))
    print("  per tier:", json.dumps(best["scores"]["recall"][5]))
    print("  by year: ", json.dumps(best["scores"]["by_year"]))
    print()
    print(f"wrote {out_path}")
    if len(configs) == 1:
        print(f"      {args.index}")


if __name__ == "__main__":
    main()
