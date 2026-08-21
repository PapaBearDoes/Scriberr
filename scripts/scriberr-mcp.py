#!/usr/bin/env python3
"""
scriberr-mcp.py -- MCP server exposing the podcast corpus to Claude.

RUNS INSIDE WSL, launched by Claude Desktop via wsl.exe. Deliberately not a
Windows process:
  - it reuses the existing venv rather than needing a second Python install
  - **the fallback path stays fast.** A Windows-side server falling back to
    BM25 would read the index over SMB; from here it reads the LOCAL copy at
    ~45 ms instead of ~286 ms. Measured 21 Aug on the same 31,434-chunk file.
  - one filesystem, no Y:\\ translation

STDOUT IS THE MCP TRANSPORT. A single stray print() corrupts the JSON-RPC
stream and the server dies in a way that looks like "the tool never appeared".
Everything diagnostic goes to stderr. There are no bare prints in this file and
there should never be.

ARCHITECTURE. The sidecar (scriberr-rerank-sidecar.service) holds
bge-reranker-v2-m3 resident and does both FTS5 search and reranking, ~600 ms
per query against ~6,800 ms for the CLI. This server is a thin HTTP client to
it. When the sidecar does not answer inside SIDECAR_TIMEOUT -- it is down, or
the GPU is saturated by gaming or the transcription backfill -- this falls back
to querying the local index directly and returns BM25 order.

**THE FALLBACK IS ALWAYS DISCLOSED IN THE RESULT.** An agent that got unranked
results because the GPU was busy has to be able to say so. Silently degrading
quality is worse than being slow.

REGISTER IT (claude_desktop_config.json):
    "scriberr": {
      "command": "wsl.exe",
      "args": ["-d", "Debian", "-e",
               "/home/hermes/venvs/rerank/bin/python",
               "/home/hermes/scriberr/scripts/scriberr-mcp.py"]
    }

Confirm the distro name with `wsl -l -q` first.

REQUIRES the `mcp` package in the venv:
    ~/venvs/rerank/bin/pip install mcp
"""

import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.request

SIDECAR = os.environ.get("SCRIBERR_SIDECAR", "http://127.0.0.1:8422")
LOCAL_INDEX = os.environ.get(
    "SCRIBERR_INDEX", os.path.expanduser("~/.local/share/scriberr/chunks.sqlite"))
FALLBACK_INDEX = "/storage/nas/ai/scriberr/index/chunks.sqlite"

# Measured idle: total p95 716 ms, so 2x p95 is ~1,500 ms. Set a little above
# that because a warm-but-busy GPU can legitimately overshoot, and falling back
# costs real ranking quality (R@1 0.506 -> 0.628 with reranking). Falling back
# is cheap and fast, so the cost of waiting slightly too long is small and the
# cost of giving up slightly too early is not.
SIDECAR_TIMEOUT = float(os.environ.get("SCRIBERR_TIMEOUT", "2.0"))

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def err(msg):
    """stderr only -- see the module docstring."""
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------- sidecar

def sidecar_post(path, payload, timeout=SIDECAR_TIMEOUT):
    req = urllib.request.Request(
        SIDECAR + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sidecar_get(path, timeout=SIDECAR_TIMEOUT):
    with urllib.request.urlopen(SIDECAR + path, timeout=timeout) as r:
        return json.loads(r.read())


# ------------------------------------------------------------------ fallback

def index_path():
    return LOCAL_INDEX if os.path.exists(LOCAL_INDEX) else FALLBACK_INDEX


def fts_expr(text, min_len=2):
    """Must match the sidecar and search-corpus.py: OR'd terms, minimum length
    2. Length 3 silently discards "ai"; 4 also loses "seo", "cta", "roi"."""
    toks = [t for t in WORD_RE.findall((text or "").lower()) if len(t) >= min_len]
    if not toks:
        return None
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks[:60])


def clock(seconds):
    s = int(seconds or 0)
    return f"{s // 60}:{s % 60:02d}"


def bound(v, high):
    if not v:
        return None
    v = str(v).strip()
    if len(v) == 8:
        return v
    if len(v) == 4:
        return v + ("1231" if high else "0101")
    raise ValueError(f"date must be YYYY or YYYYMMDD, got {v!r}")


def local_search(query, k, show, since, until, max_per_episode):
    """BM25 only, used when the sidecar does not answer. Deliberately duplicated
    from search-corpus.py rather than imported: that module's filename contains a
    hyphen so a plain import is impossible, and importlib gymnastics in the
    fallback path would be a worse failure mode than thirty repeated lines."""
    expr = fts_expr(query)
    if not expr:
        raise ValueError("query has no searchable terms")
    lo, hi = bound(since, False), bound(until, True)
    shows = [s.strip() for s in show.split(",")] if show else None

    db = sqlite3.connect(f"file:{index_path()}?mode=ro", uri=True)
    try:
        sql = ("SELECT text_for_model, stem, upload_date, start_s, end_s, show"
               " FROM chunks WHERE chunks MATCH ?")
        p = [expr]
        if shows:
            sql += " AND show IN (" + ",".join("?" * len(shows)) + ")"
            p.extend(shows)
        if lo:
            sql += " AND upload_date >= ?"
            p.append(lo)
        if hi:
            sql += " AND upload_date <= ?"
            p.append(hi)
        sql += " ORDER BY bm25(chunks) LIMIT ?"
        p.append(max(k * 8, 50))
        rows = db.execute(sql, p).fetchall()
    finally:
        db.close()

    picked, per_ep = [], {}
    for r in rows:
        if max_per_episode and per_ep.get(r[1], 0) >= max_per_episode:
            continue
        per_ep[r[1]] = per_ep.get(r[1], 0) + 1
        picked.append(r)
        if len(picked) >= k:
            break
    return [{"show": r[5], "upload_date": r[2],
             "timestamp": f"{clock(r[3])}-{clock(r[4])}", "text": r[0]}
            for r in picked]


# -------------------------------------------------------------- formatting

def render(payload, results, note=None):
    lines = []
    if note:
        lines.append(note)
    if not results:
        lines.append("No passages matched. Try fewer or more common words, "
                     "widen the dates, or drop the show filter.")
        return "\n".join(lines)

    idx = payload.get("index") or {}
    lines.append(
        f"{len(results)} passage(s). Ranking: "
        + ("cross-encoder reranked." if payload.get("reranked")
           else f"BM25 keyword order only ({payload.get('rerank_reason') or 'unranked'}).")
        + (f" Index built {idx.get('built')}." if idx.get("built") else ""))
    lines.append("")
    for i, r in enumerate(results, 1):
        d = r.get("upload_date") or ""
        pretty = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else (d or "undated")
        lines.append(f"### {i}. {r.get('show')} — {pretty} — {r.get('timestamp')}")
        # text_for_model already leads with show | title | episode | date. These
        # are TACTICAL podcasts and 2018 martech advice predates GA4, iOS ATT
        # and LLMs -- the date travelling with the text is what stops stale
        # tactics being quoted as current.
        lines.append(r.get("text", "").strip())
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------- server

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    err("ABORT: the `mcp` package is not installed in this interpreter.\n"
        "       ~/venvs/rerank/bin/pip install mcp")
    sys.exit(1)

mcp = FastMCP("scriberr")


@mcp.tool()
def search_podcasts(query: str, k: int = 6, show: str = "", since: str = "",
                    until: str = "", max_per_episode: int = 2,
                    max_per_show: int = 0) -> str:
    """Search transcribed marketing podcasts for passages relevant to a query.

    Returns verbatim passages with their show, episode, date and timestamp, so
    they can be cited. The corpus is ~2,600 episodes of tactical marketing and
    martech advice spanning 2018 to now.

    DATES MATTER MORE THAN USUAL HERE. These shows give platform-specific
    tactics, and advice from 2018 predates GA4, iOS ATT and LLMs entirely. Use
    `since` when you want what currently works; leave it off when you want to
    see how a tactic has aged.

    Args:
        query: what you want to know, in natural language.
        k: passages to return, 1-20. Default 6.
        show: comma-separated show names to restrict to. Call
            list_podcast_shows first -- a wrong name returns an error listing
            the valid ones.
        since: earliest upload date, "2025" or "20250101".
        until: latest upload date, same format.
        max_per_episode: cap per episode so one episode cannot fill the
            results. 0 disables. Default 2.
        max_per_show: cap per show. One show may outnumber another 4:1 in the
            index, and keyword ranking has no notion that source diversity
            matters. 0 (default) lets relevance decide.
    """
    k = max(1, min(int(k), 20))
    req = {"query": query, "k": k, "max_per_episode": int(max_per_episode),
           "max_per_show": int(max_per_show), "rerank": True}
    for name, val in (("show", show), ("since", since), ("until", until)):
        if val:
            req[name] = val

    try:
        payload = sidecar_post("/search", req)
        return render(payload, payload.get("results", []))
    except urllib.error.HTTPError as e:
        # 400 is a real user error -- an unknown show name, an unparseable date.
        # Surfacing it beats silently falling back to a query with the bad
        # filter dropped, which would return plausible but wrong results.
        try:
            detail = json.loads(e.read()).get("error", str(e))
        except Exception:
            detail = str(e)
        return f"Search rejected: {detail}"
    except Exception as exc:
        err(f"sidecar unavailable ({type(exc).__name__}: {exc}), using BM25 fallback")
        try:
            results = local_search(query, k, show, since, until, int(max_per_episode))
        except ValueError as ve:
            return f"Search rejected: {ve}"
        except Exception as fe:
            return (f"Search failed: the rerank service is unreachable and the "
                    f"local index could not be queried ({type(fe).__name__}: {fe}).")
        return render({"reranked": False,
                       "rerank_reason": "rerank service unreachable"},
                      results,
                      note="**Reranking unavailable — these are keyword-ranked "
                           "results and the ordering is weaker than usual.**")


@mcp.tool()
def list_podcast_shows() -> str:
    """List the podcasts in the index, with episode counts and date ranges.

    Call this before using the `show` filter on search_podcasts -- show names
    are exact directory names, not display titles, and they change as shows are
    added.
    """
    try:
        health = sidecar_get("/health")
        shows = health.get("shows", [])
        idx = health.get("index", {})
        head = (f"{idx.get('chunks', 0):,} passages, index built "
                f"{idx.get('built')}.")
    except Exception:
        db = sqlite3.connect(f"file:{index_path()}?mode=ro", uri=True)
        try:
            shows = [r[0] for r in
                     db.execute("SELECT DISTINCT show FROM chunks ORDER BY show")]
        finally:
            db.close()
        head = "(rerank service unreachable; read directly from the index)"

    db = sqlite3.connect(f"file:{index_path()}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT show, COUNT(DISTINCT stem), MIN(upload_date), MAX(upload_date)"
            " FROM chunks GROUP BY show ORDER BY show").fetchall()
    finally:
        db.close()

    lines = [head, ""]
    for name, eps, lo, hi in rows:
        def pretty(d):
            return f"{d[:4]}-{d[4:6]}-{d[6:]}" if d and len(d) == 8 else (d or "?")
        lines.append(f"- **{name}** — {eps:,} episodes, {pretty(lo)} to {pretty(hi)}")
    if not rows:
        lines.append("(no shows found in the index)")
    return "\n".join(lines)


if __name__ == "__main__":
    err(f"scriberr-mcp: sidecar {SIDECAR}, index {index_path()}, "
        f"timeout {SIDECAR_TIMEOUT}s")
    mcp.run()
