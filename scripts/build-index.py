#!/usr/bin/env python3
"""
build-index.py -- chunk the corpus, index it with SQLite FTS5, and score the
result against the labelled eval set. One pass, one number.

WHY FTS5 AND NOT A VECTOR STORE. Measured 17 Aug on 1,200 queries over Do This
NOT That: R@1 0.583, R@5 0.785, R@10 0.832, R@25 0.872, R@50 0.903, R@100 0.933.
BM25 has the right chunk in its top 50 for 90% of queries -- it ORDERS them
badly, it is not failing to FIND them. Only 6.75% never surface in 100 results,
and that is the entire theoretical headroom for a different retrieval method.
The decision was a cross-encoder reranker over the top 50, not embeddings.

MULTI-SHOW SINCE 2026-08-19. The podcast tree is per show on the NAS:

    <root>/transcripts/<Show>/<stem>.json
    <root>/audio/<Show>/<stem>.info.json      sidecars live WITH the audio

ONE INDEX HOLDS EVERY SHOW, with a `show` column and a `--show` filter, rather
than an index per show. Cross-show retrieval is the point: the best passage on
a tactic may come from any source. The cost is that each added show adds
distractors, so per-show recall drops as the corpus grows. That drop is real
information about the retrieval task, not noise -- which is why recall is
reported per show as well as overall.

THE EVAL SET IS NOT A TRAINING SET -- READ THIS BEFORE CHANGING THE CHUNKER.
Tier-1 queries ARE the publisher's Best Moments descriptions, and those same
timestamps are what the `moments` chunker splits on. Splitting is fine: a
boundary carries no text. Writing the descriptions INTO the chunk body would
not be -- it would index the answer key and every score would be garbage.
Nothing here does that. Keep it that way.

For the same reason the episode title and date go in an UNINDEXED column by
default. Publisher descriptions echo the title constantly, so indexing it
inflates recall without improving retrieval. `--index-header` measures that
effect; it is not meant to be left on.

USAGE
    python3 scripts/build-index.py                    # build + score, defaults
    python3 scripts/build-index.py --sweep            # chunking grid
    python3 scripts/build-index.py --sweep-query      # query-side grid
    python3 scripts/build-index.py --dump-misses      # why the misses missed
    python3 scripts/build-index.py --eval-sample 1200 # faster iteration
    python3 scripts/build-index.py --query "subject lines"

OUTPUTS
    <analysis>/index-eval.json   every configuration scored
    <index>                      chunks.sqlite, built locally then copied

SQLITE ON NFS: the index is built in a local temp file and copied to its final
location. SQLite locking over NFS is a hazard and a half-written index that
reports success is the exact failure this project keeps hitting.
"""

import argparse
import html
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

DEFAULT_ROOT = "/storage/nas/ai/scriberr/podcasts"
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

# Function words carry almost no retrieval signal but each becomes an OR clause.
# MEASURED SLIGHTLY NEGATIVE (0.785 -> 0.780), so filtering is OFF by default.
# Minimum term length matters far more: len>=3 silently discards "ai", len>=4
# also loses "seo", "cta" and "roi" -- fatal in a marketing corpus.
STOPWORDS = set("""
a about above after again against all also am an and any are aren as at be
because been before being below between both but by can cannot could couldn did
didn do does doesn doing don down during each few for from further had hadn has
hasn have haven having he her here hers herself him himself his how however i
if in into is isn it its itself just let me more most much must my myself no
nor not now of off on once only or other ought our ours ourselves out over own
same shan she should shouldn so some such than that the their theirs them
themselves then there these they this those through to too under until up very
was wasn we were weren what when where whether which while who whom why will
with won would wouldn you your yours yourself yourselves
""".split())


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


def query_terms(text, use_stopwords, min_len):
    toks = [t for t in WORD_RE.findall((text or "").lower()) if len(t) >= min_len]
    if use_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks[:60]


def overlap_fraction(terms, text):
    ts = set(terms)
    if not ts:
        return 0.0, []
    low = (text or "").lower()
    present = sorted(t for t in ts if t in low)
    return len(present) / len(ts), present


def build_df(rows):
    df = Counter()
    for r in rows:
        for t in set(WORD_RE.findall(r["body"].lower())):
            df[t] += 1
    return df


def overlap_weighted(terms, text, df, n_chunks):
    """
    IDF-weighted share of query terms present, for the false-miss proxy.

    THE FLAT VERSION WAS RIGGED: BM25 selects the top 5 BECAUSE they contain
    query terms; the labelled target is not selected that way, so comparing
    them on term overlap favours the selection almost regardless of quality.
    It returned 86%. Weighting by rarity narrowed it only to 83.3%, so the bias
    largely survives. STILL A PROXY -- the deep-k recall curve is the unbiased
    instrument and the decision rests on that, not on this.
    """
    ts = set(terms)
    if not ts:
        return 0.0
    low = (text or "").lower()
    total = got = 0.0
    for t in ts:
        w = math.log((n_chunks + 1) / (df.get(t, 0) + 1)) + 1.0
        total += w
        if t in low:
            got += w
    return got / total if total else 0.0


# ---------------------------------------------------------------- corpus load

def discover_shows(root, only=None):
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
        sys.exit(f"ABORT: no show directories under {troot}"
                 + (f" matching {sorted(only)}" if only else ""))
    return shows


def load_corpus(root, only=None, limit=0):
    """Read every episode once. Held in memory so a sweep does not re-read
    thousands of files per configuration."""
    episodes = []
    for show, tdir, adir in discover_shows(root, only):
        stems = sorted(f[:-5] for f in os.listdir(tdir) if f.endswith(".json"))
        if limit:
            stems = stems[:limit]
        if not stems:
            print(f"  {show}: no transcripts, skipping", file=sys.stderr)
            continue
        for stem in stems:
            try:
                with open(os.path.join(tdir, stem + ".json"),
                          encoding="utf-8", errors="replace") as fh:
                    tj = json.load(fh)
            except Exception as exc:
                print(f"  skipping {show}/{stem}: {type(exc).__name__}", file=sys.stderr)
                continue
            side = {}
            spath = os.path.join(adir, stem + ".info.json")
            if os.path.exists(spath):
                try:
                    with open(spath, encoding="utf-8", errors="replace") as fh:
                        side = json.load(fh)
                except Exception:
                    pass
            episodes.append({
                "show": show,
                "stem": stem,
                "title": tj.get("title") or side.get("title") or stem,
                "episode_number": side.get("episode_number"),
                "upload_date": side.get("upload_date"),
                "segments": (tj.get("transcript") or {}).get("segments") or [],
                "moments": parse_moments(side.get("description", "")),
            })
        print(f"  {show}: {sum(1 for e in episodes if e['show'] == show)} episodes",
              file=sys.stderr)
    return episodes


# ----------------------------------------------------------------- boilerplate

def build_stoplists(episodes, floor, fraction):
    """
    Derive boilerplate FROM THE CORPUS, PER SHOW, never from a hand-written list.

    Per show because a flat threshold cannot serve a 636-episode show and a
    30-episode one at once: 20 is 3% of the first and 67% of the second.

    From the corpus because handwritten lists miss variants. Do This NOT That's
    intro exists as at least three: `welcome to do this not that the podcast FOR
    marketers` (278 episodes) and `...FROM marketers` (177) are the same spoken
    line landing two ways. A list catches one and silently misses the other.

    Frequency across distinct episodes is the whole rule -- position is not used,
    because it would miss the mid-roll cross-promo sitting ~75% through the ~112
    Bathroom Break episodes. A sentence appearing verbatim in that many separate
    episodes is boilerplate whatever position it occupies.
    """
    out = {}
    by_show = defaultdict(list)
    for e in episodes:
        by_show[e["show"]].append(e)
    for show, eps in by_show.items():
        if floor <= 0:                      # disabled
            out[show] = set()
            continue
        seen = Counter()
        for e in eps:
            for norm in {normalize(s.get("text")) for s in e["segments"]}:
                if len(norm) >= 15:
                    seen[norm] += 1
        thresh = max(floor, int(math.ceil(fraction * len(eps))))
        out[show] = {t for t, c in seen.items() if c >= thresh}
    return out


# -------------------------------------------------------------------- chunking

def split_oversized(group, max_words):
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
        if out and n < max_words * 0.4:     # don't leave a stub
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
    # The header exists so the CONSUMING model sees WHICH SHOW and WHEN. These
    # are tactical podcasts: undated 2018 martech advice reads as current, and
    # with several shows indexed together the source matters too.
    header = f"{ep['show']} | {ep['title']} | episode {ep['episode_number']} | {pretty}"

    rows = []
    for g in groups:
        body = " ".join((s.get("text") or "").strip() for s in g).strip()
        if not body:
            continue
        rows.append({
            "show": ep["show"],
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
            show UNINDEXED,
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
        "INSERT INTO chunks (body, text_for_model, header, show, stem,"
        " episode_number, upload_date, start_s, end_s, speakers)"
        " VALUES (:body,:text_for_model,:header,:show,:stem,"
        " :episode_number,:upload_date,:start_s,:end_s,:speakers)",
        rows,
    )
    db.commit()
    return db


def fts_query(text, use_stopwords=False, min_len=2):
    """
    Free text -> a safe FTS5 MATCH expression.

    Bare multi-word MATCH is implicit AND, which returns nothing for a 20-word
    publisher description. OR plus bm25 ranking is the right shape: every term
    contributes, rare terms dominate. Each token is quoted so apostrophes and
    FTS operators cannot break the parse.
    """
    toks = query_terms(text, use_stopwords, min_len)
    if not toks:
        return None
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks)


def search(db, query, k, use_stopwords=False, min_len=2, show=None):
    expr = fts_query(query, use_stopwords, min_len)
    if not expr:
        return []
    sql = ("SELECT show, stem, start_s, end_s, bm25(chunks) AS score, body"
           " FROM chunks WHERE chunks MATCH ?")
    params = [expr]
    if show:
        sql += " AND show = ?"
        params.append(show)
    sql += " ORDER BY score LIMIT ?"
    params.append(k)
    return db.execute(sql, params).fetchall()


# ------------------------------------------------------------------ evaluation

def target_text(db, row, terms=()):
    """(all_target_text, best_excerpt) for the row's known-correct location.
    For tier-2 the target is the whole episode, so the excerpt is the chunk with
    the most query-term overlap -- NOT the first, which used to make every
    tier-2 excerpt print the episode intro and look like a vocabulary problem
    that was not there."""
    if row["tier"] == "episode":
        rows = [r[0] for r in db.execute(
            "SELECT body FROM chunks WHERE show = ? AND stem = ?",
            (row["show"], row["stem"])).fetchall()]
    else:
        t = row["t_seconds"]
        rows = [r[0] for r in db.execute(
            "SELECT body FROM chunks WHERE show = ? AND stem = ?"
            " AND end_s >= ? AND start_s <= ?",
            (row["show"], row["stem"], t - HIT_BEFORE, t + HIT_AFTER)).fetchall()]
    if not rows:
        return "", ""
    joined = " ".join(rows)
    best = max(rows, key=lambda b: overlap_fraction(terms, b)[0]) if terms else rows[0]
    return joined, best


def evaluate(db, eval_rows, ks, use_stopwords=False, min_len=2,
             collect_misses=False, df=None, n_chunks=0):
    maxk = max(ks)
    hits = {k: Counter() for k in ks}
    totals = Counter()
    show_totals = Counter()
    show_hits = Counter()
    rr = defaultdict(list)
    by_year = defaultdict(lambda: {"n": 0, "hit": 0})
    misses = []
    overlap_hist = Counter()

    for row in eval_rows:
        tier = row["tier"]
        totals[tier] += 1
        show_totals[row["show"]] += 1
        year = (row.get("upload_date") or "????")[:4]
        by_year[year]["n"] += 1

        results = search(db, row["publisher_description"], maxk, use_stopwords, min_len)
        rank = None
        for i, (show, stem, start_s, end_s, _score, _body) in enumerate(results, 1):
            # MATCH ON (show, stem), NOT stem alone. Two shows can publish
            # episodes whose sanitised filenames collide, and a cross-show
            # false positive would silently inflate recall.
            if show != row["show"] or stem != row["stem"]:
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
                show_hits[row["show"]] += 1

        if collect_misses and (rank is None or rank > 5):
            terms = query_terms(row["publisher_description"], use_stopwords, min_len)
            tgt_all, tgt_best = target_text(db, row, terms)
            tgt_frac, present = overlap_fraction(terms, tgt_all)
            tgt_w = overlap_weighted(terms, tgt_all, df or {}, n_chunks)

            best_res_frac, best_res, best_res_w = 0.0, None, 0.0
            for r in results[:5]:
                f, _ = overlap_fraction(terms, r[5])
                if f > best_res_frac:
                    best_res_frac, best_res = f, r
                best_res_w = max(best_res_w, overlap_weighted(terms, r[5], df or {}, n_chunks))

            overlap_hist[min(len(present), 10)] += 1
            top = results[0] if results else None
            misses.append({
                "tier": tier, "rank": rank,
                "not_in_top_k_max": rank is None,
                "likely_false_miss": bool(terms) and best_res_frac >= tgt_frac > 0,
                "likely_false_miss_weighted": bool(terms) and best_res_w >= tgt_w > 0,
                "show": row["show"], "stem": row["stem"],
                "upload_date": row.get("upload_date"),
                "t_seconds": row.get("t_seconds"),
                "query": row["publisher_description"],
                "n_terms": len(set(terms)),
                "terms_present_in_target": present,
                "target_overlap_fraction": round(tgt_frac, 3),
                "target_overlap_weighted": round(tgt_w, 3),
                "best_result_overlap_fraction": round(best_res_frac, 3),
                "best_result_overlap_weighted": round(best_res_w, 3),
                "target_found": bool(tgt_all.strip()),
                "target_excerpt": tgt_best[:400],
                "top_result_show": top[0] if top else None,
                "top_result_stem": top[1] if top else None,
                "best_result_show": best_res[0] if best_res else None,
                "best_result_excerpt": (best_res[5][:400] if best_res else None),
            })

    out = {"totals": dict(totals), "recall": {}, "mrr": {}, "by_year": {}, "by_show": {}}
    all_n = sum(totals.values())
    for k in ks:
        out["recall"][k] = {t: round(hits[k][t] / totals[t], 4) for t in totals if totals[t]}
        out["recall"][k]["all"] = round(sum(hits[k].values()) / all_n, 4) if all_n else 0
    for t, vals in rr.items():
        out["mrr"][t] = round(statistics.fmean(vals), 4) if vals else 0
    for y, d in sorted(by_year.items()):
        out["by_year"][y] = {"n": d["n"], "recall@5": round(d["hit"] / d["n"], 4) if d["n"] else 0}
    for s, n in sorted(show_totals.items()):
        out["by_show"][s] = {"n": n, "recall@5": round(show_hits[s] / n, 4) if n else 0}

    if collect_misses:
        n_miss = len(misses) or 1
        total_q = all_n or 1
        false_w = [m for m in misses if m["likely_false_miss_weighted"]]
        out["miss_analysis"] = {
            "n_misses_at_5": len(misses),
            "n_not_in_top_k_max": sum(1 for m in misses if m["not_in_top_k_max"]),
            "k_max": maxk,
            "n_ranked_below_5_but_found": sum(1 for m in misses if not m["not_in_top_k_max"]),
            "overlap_histogram_terms_present": dict(sorted(overlap_hist.items())),
            "zero_overlap_pct": round(100.0 * sum(1 for m in misses if not m["terms_present_in_target"]) / n_miss, 1),
            "high_overlap_pct": round(100.0 * sum(1 for m in misses if m["target_overlap_fraction"] >= 0.6) / n_miss, 1),
            "target_missing_from_index_pct": round(100.0 * sum(1 for m in misses if not m["target_found"]) / n_miss, 1),
            "likely_false_miss_pct_FLAT_BIASED": round(100.0 * sum(1 for m in misses if m["likely_false_miss"]) / n_miss, 1),
            "likely_false_miss_pct_idf_weighted": round(100.0 * len(false_w) / n_miss, 1),
            "adjusted_recall_at_5_upper_bound": round(
                out["recall"][5].get("all", 0) + len(false_w) / total_q, 4),
            "vague_queries_pct": round(100.0 * sum(1 for m in misses if m["n_terms"] <= 4) / n_miss, 1),
        }
    return out, misses


# ---------------------------------------------------------------------- driver

def run_config(episodes, eval_rows, cfg, ks, index_path=None, collect_misses=False):
    stoplists = build_stoplists(episodes, cfg["boiler_floor"], cfg["boiler_fraction"])
    rows = []
    for ep in episodes:
        rows.extend(chunk_episode(ep, cfg, stoplists.get(ep["show"], set())))
    if not rows:
        sys.exit("ABORT: chunker produced no rows")

    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    db = build_index(tmp.name, rows)
    df = build_df(rows) if collect_misses else None
    try:
        scores, misses = evaluate(db, eval_rows, ks, cfg["stopwords"],
                                  cfg["min_term_len"], collect_misses, df, len(rows))
    finally:
        db.close()

    wc = [words(r["body"]) for r in rows]
    per_show = Counter(r["show"] for r in rows)
    result = {
        "config": cfg,
        "stoplist_sizes": {s: len(v) for s, v in stoplists.items()},
        "n_chunks": len(rows),
        "chunks_per_show": dict(per_show),
        "chunk_words": {
            "median": round(statistics.median(wc), 1) if wc else 0,
            "mean": round(statistics.fmean(wc), 1) if wc else 0,
            "max": max(wc) if wc else 0,
        },
        "scores": scores,
    }
    if index_path:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        shutil.copy2(tmp.name, index_path)      # build local, copy to NFS
        result["index_path"] = index_path
    os.unlink(tmp.name)
    return result, misses


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--show", default=None, help="comma-separated show names; default all")
    ap.add_argument("--limit", type=int, default=0, help="episodes PER SHOW (smoke test)")
    ap.add_argument("--eval-sample", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="chunking grid")
    ap.add_argument("--sweep-query", action="store_true", help="query-side grid")
    ap.add_argument("--dump-misses", nargs="?", const="AUTO", default=None)
    ap.add_argument("--query", help="search the built index and print results")
    ap.add_argument("--mode", default="moments", choices=["moments", "window"])
    ap.add_argument("--target-words", type=int, default=250)
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--boiler-floor", type=int, default=10,
                    help="min episodes a line must appear in; 0 disables stripping")
    ap.add_argument("--boiler-fraction", type=float, default=0.05,
                    help="or this share of the show's episodes, whichever is larger")
    ap.add_argument("--index-header", action="store_true",
                    help="also index show/title/date (inflates recall, see docstring)")
    ap.add_argument("--stopwords", action="store_true",
                    help="drop function words; MEASURED SLIGHTLY WORSE, off by default")
    ap.add_argument("--min-term-len", type=int, default=2,
                    help="2 is the measured best; 3 discards 'ai', 4 also 'seo'/'cta'/'roi'")
    args = ap.parse_args()

    only = {s.strip() for s in args.show.split(",")} if args.show else None
    print("loading corpus ...", file=sys.stderr)
    episodes = load_corpus(args.root, only, args.limit)
    if not episodes:
        sys.exit("ABORT: no episodes loaded")

    eval_path = os.path.join(args.analysis, "eval-candidates.jsonl")
    if not os.path.exists(eval_path):
        sys.exit(f"ABORT: no eval set at {eval_path} -- run measure-corpus.py first")
    keys = {(e["show"], e["stem"]) for e in episodes}
    eval_rows = []
    with open(eval_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "show" not in r:
                sys.exit("ABORT: eval-candidates.jsonl has no `show` field -- it was\n"
                         "       written by a pre-2026-08-19 measure-corpus.py.\n"
                         "       Re-run: python3 scripts/measure-corpus.py")
            if (r["show"], r["stem"]) in keys:
                eval_rows.append(r)
    if args.eval_sample and args.eval_sample < len(eval_rows):
        step = len(eval_rows) / args.eval_sample     # deterministic, spread across shows
        eval_rows = [eval_rows[int(i * step)] for i in range(args.eval_sample)]
    print(f"  {len(eval_rows)} eval queries", file=sys.stderr)

    base = {
        "mode": args.mode, "target_words": args.target_words,
        "max_words": args.max_words, "overlap": args.overlap,
        "boiler_floor": args.boiler_floor, "boiler_fraction": args.boiler_fraction,
        "index_header": args.index_header,
        "stopwords": args.stopwords, "min_term_len": args.min_term_len,
    }
    # DEEP k SETTLES RANKING VERSUS RETRIEVAL, and unlike the false-miss proxy it
    # is unbiased. A steep rise from R@10 to R@50 means BM25 is FINDING the right
    # chunks and ordering them badly -- fix with a reranker over BM25 candidates,
    # not a second retrieval system.
    ks = [1, 3, 5, 10, 25, 50, 100]

    if args.query:
        run_config(episodes, [], base, ks, index_path=args.index)
        db = sqlite3.connect(args.index)
        print()
        for show, stem, start_s, end_s, score, _b in search(
                db, args.query, 8, base["stopwords"], base["min_term_len"],
                show=(sorted(only)[0] if only and len(only) == 1 else None)):
            print(f"  {score:8.2f}  {int(start_s):5d}s  {show:<16} {stem[:60]}")
        db.close()
        return

    configs = [base]
    if args.sweep:
        configs = [{**base, "mode": m, "target_words": tw, "boiler_floor": bf}
                   for m in ("moments", "window")
                   for tw in (200, 250, 350)
                   for bf in (0, 10)]
    elif args.sweep_query:
        # Chunking made no measurable difference across 12 configurations on
        # 17 Aug, so it is pinned here and the query side varies instead.
        configs = [{**base, "stopwords": sw, "min_term_len": ml}
                   for sw in (False, True) for ml in (2, 3, 4)]

    collect = args.dump_misses is not None
    results, all_misses = [], []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg['mode']} tw={cfg['target_words']} "
              f"bf={cfg['boiler_floor']} sw={cfg['stopwords']} "
              f"ml={cfg['min_term_len']} ...", file=sys.stderr)
        keep = args.index if len(configs) == 1 else None
        res, misses = run_config(episodes, eval_rows, cfg, ks, keep, collect)
        results.append(res)
        if collect and not all_misses:
            all_misses = misses                 # first configuration only

    os.makedirs(args.analysis, exist_ok=True)
    out_path = os.path.join(args.analysis, "index-eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print()
    print(f"{'mode':<8} {'tw':>4} {'bf':>3} {'sw':>3} {'ml':>3} {'chunks':>7} "
          f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@25':>7} {'R@50':>7} {'R@100':>7} {'MRR-c':>7}")
    for r in results:
        c, s = r["config"], r["scores"]
        print(f"{c['mode']:<8} {c['target_words']:>4} {c['boiler_floor']:>3} "
              f"{str(c['stopwords'])[0]:>3} {c['min_term_len']:>3} {r['n_chunks']:>7} "
              + " ".join(f"{s['recall'][k].get('all', 0):>7.4f}" for k in (1, 5, 10, 25, 50, 100))
              + f" {s['mrr'].get('chunk', 0):>7.4f}")

    best = max(results, key=lambda r: r["scores"]["recall"][5].get("all", 0))
    print()
    print("best by recall@5:", json.dumps(best["config"]))
    print("  per tier: ", json.dumps(best["scores"]["recall"][5]))
    print("  per show: ", json.dumps(best["scores"]["by_show"]))
    print("  chunks:   ", json.dumps(best["chunks_per_show"]))
    print("  by year:  ", json.dumps(best["scores"]["by_year"]))

    if collect and all_misses:
        ma = results[0]["scores"].get("miss_analysis", {})
        path = args.dump_misses if args.dump_misses != "AUTO" \
            else os.path.join(args.analysis, "misses.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for m in all_misses:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        print()
        print(f"MISS ANALYSIS  (config 1 of {len(configs)}, {ma.get('n_misses_at_5')} misses at k=5)")
        print(f"  found but ranked >5                  {ma.get('n_ranked_below_5_but_found')}"
              f"   (never in top {ma.get('k_max')}: {ma.get('n_not_in_top_k_max')})")
        print(f"  LIKELY FALSE MISSES, flat (BIASED)   {ma.get('likely_false_miss_pct_FLAT_BIASED')}%")
        print(f"  LIKELY FALSE MISSES, idf-weighted    {ma.get('likely_false_miss_pct_idf_weighted')}%"
              "   <- narrower bias, still a proxy")
        print(f"  zero term overlap with the target    {ma.get('zero_overlap_pct')}%"
              "   <- vocabulary mismatch, the only case embeddings uniquely fix")
        print(f"  60%+ overlap and still missed        {ma.get('high_overlap_pct')}%"
              "   <- ranking failure, a reranker is cheaper")
        print(f"  target absent from the index         {ma.get('target_missing_from_index_pct')}%")
        print(f"  vague queries (<=4 terms)            {ma.get('vague_queries_pct')}%")
        print()
        print(f"  recall@5 measured                    {results[0]['scores']['recall'][5].get('all', 0)}   <- lower bound")
        print(f"  recall@5 adjusted for false misses   {ma.get('adjusted_recall_at_5_upper_bound')}   <- UPPER bound, proxy")
        print(f"  wrote {path}")

    print()
    print(f"wrote {out_path}")
    if len(configs) == 1:
        print(f"      {args.index}")


if __name__ == "__main__":
    main()
