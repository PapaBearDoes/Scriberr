#!/usr/bin/env python3
"""
rerank-sidecar.py -- HTTP search over the podcast corpus, model held resident.

WHY A SERVICE. The CLI loads bge-reranker-v2-m3 on every invocation: measured
4.2-8.7 s. Holding it resident makes a query ~575 ms instead of ~6,800 ms. This
process does BOTH the FTS5 search and the rerank, because the index it reads is
LOCAL -- measured 21 Aug, BM25 fetch p50 45 ms local against 286 ms over NFS on
the same file. Putting the search next to the GPU rather than across a share is
most of the win; the model being resident is the rest.

    fetch      45 ms      rerank  ~520 ms warm, ~560 ms after idle
    total     ~575 ms     suggested client timeout ~1500 ms (2x p95)

KEEP-WARM IS NOT OPTIONAL. Measured back-to-back reranks ran 291 ms; with 20 s
gaps the same work took 563 ms. The CUDA context goes cold between requests and
agent traffic is bursty, so a dummy inference every --keepwarm seconds holds it
hot. Without it the resident model is worth much less than it looks.

THE INDEX IS REPLACED UNDER US. rebuild-index.sh publishes by atomic rename, so
an open SQLite handle keeps reading the OLD inode forever -- the service would
serve a stale corpus indefinitely while the file on disk was current. Every
request stats the path and reopens when it changed. That check costs microseconds
and is the difference between "fresh nightly" and "fresh until the first
rebuild".

API
    GET  /health                    index stats, model, valid show names
    POST /search  {"query": "...", "k": 6, "depth": 50, "rerank": true,
                   "show": "MarTech", "since": "2025", "until": "20260101",
                   "max_per_episode": 2, "max_per_show": 0}

Every response carries `reranked` and, when false, `rerank_reason` -- a client
that got BM25 order because the GPU was busy needs to be able to say so.

Results return `text_for_model`, which leads with show, title, episode number
and date. These are TACTICAL podcasts: 2018 martech advice predates GA4, iOS ATT
and LLMs, and an undated passage reads as current. The date travelling with the
text is what stops stale tactics being quoted as live ones.

RUN IT
    ~/venvs/rerank/bin/python scripts/rerank-sidecar.py
    curl -s localhost:8422/health | jq .
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_INDEX = os.path.expanduser("~/.local/share/scriberr/chunks.sqlite")
FALLBACK_INDEX = "/storage/nas/ai/scriberr/index/chunks.sqlite"
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
WORD_RE = re.compile(r"[A-Za-z0-9']+")

STATE = {
    "index_path": None,
    "index_mtime": 0.0,
    "chunks": 0,
    "shows": [],
    "model": None,
    "model_name": None,
    "device": "cpu",
    "loaded_at": None,
    "queries": 0,
    "reranked": 0,
    "degraded": 0,
}
GPU_LOCK = threading.Lock()      # one inference at a time; the GPU is not shared


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def fts_expr(text, min_len=2):
    """Same construction as search-corpus.py: OR'd terms, minimum length 2.
    Length 3 silently discards "ai"; 4 also loses "seo", "cta", "roi"."""
    toks = [t for t in WORD_RE.findall((text or "").lower()) if len(t) >= min_len]
    if not toks:
        return None
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks[:60])


def clock(seconds):
    s = int(seconds or 0)
    return f"{s // 60}:{s % 60:02d}"


def open_index(path):
    """Reopen if the file changed. See the docstring: atomic rename means a held
    handle reads a deleted inode forever."""
    mtime = os.path.getmtime(path)
    if mtime != STATE["index_mtime"]:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        STATE["chunks"] = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        STATE["shows"] = [r[0] for r in
                          db.execute("SELECT DISTINCT show FROM chunks ORDER BY show")]
        STATE["index_mtime"] = mtime
        STATE["index_path"] = path
        log(f"index reloaded: {STATE['chunks']:,} chunks, "
            f"shows {', '.join(STATE['shows'])}, "
            f"built {datetime.fromtimestamp(mtime, timezone.utc):%Y-%m-%d %H:%M}Z")
        return db
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


def bound(v, high):
    """'2025' -> '20250101' or '20251231'; 'YYYYMMDD' passes through."""
    if not v:
        return None
    v = str(v).strip()
    if len(v) == 8:
        return v
    if len(v) == 4:
        return v + ("1231" if high else "0101")
    raise ValueError(f"date must be YYYY or YYYYMMDD, got {v!r}")


def fetch(db, expr, shows, lo, hi, limit):
    sql = ("SELECT text_for_model, header, stem, upload_date, start_s, end_s,"
           " bm25(chunks) AS score, show, body, episode_number"
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
    sql += " ORDER BY score LIMIT ?"
    p.append(limit)
    return db.execute(sql, p).fetchall()


def do_search(req):
    query = (req.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    k = int(req.get("k", 6))
    depth = int(req.get("depth", 50))
    want_rerank = bool(req.get("rerank", True))
    max_per_ep = int(req.get("max_per_episode", 2))
    max_per_show = int(req.get("max_per_show", 0))

    shows = req.get("show")
    if isinstance(shows, str):
        shows = [s.strip() for s in shows.split(",") if s.strip()]
    if shows:
        # Show names are LOCAL KNOWLEDGE and change as shows are added. An agent
        # guessing "MarTech Podcast" deserves the list, not zero results.
        unknown = [s for s in shows if s not in STATE["shows"]]
        if unknown:
            raise ValueError(f"unknown show(s) {unknown}; valid: {STATE['shows']}")

    lo, hi = bound(req.get("since"), False), bound(req.get("until"), True)

    expr = fts_expr(query)
    if not expr:
        raise ValueError("query has no searchable terms (need words of 2+ chars)")

    db = open_index(STATE["index_path"])
    t0 = time.perf_counter()
    try:
        if max_per_show:
            # Per-show queries merged by score. A cap applied to ONE global
            # result set cannot diversify -- it can only drop results, because
            # the smaller show never reaches the global top-N at all.
            targets = shows or STATE["shows"]
            rows = []
            for s in targets:
                rows.extend(fetch(db, expr, [s], lo, hi,
                                  depth if want_rerank else max(max_per_show * 6, 20)))
            rows.sort(key=lambda r: r[6])
        else:
            rows = fetch(db, expr, shows, lo, hi,
                         depth if want_rerank else max(k * 8, 50))
    finally:
        db.close()
    fetch_ms = (time.perf_counter() - t0) * 1000

    reranked, reason, rerank_ms = False, None, 0.0
    if want_rerank and rows:
        if STATE["model"] is None:
            reason = "model not loaded"
        else:
            t = time.perf_counter()
            with GPU_LOCK:
                scores = STATE["model"].predict([(query, r[8]) for r in rows],
                                                batch_size=16, show_progress_bar=False)
            rows = [rows[j] for j in sorted(range(len(rows)), key=lambda j: -scores[j])]
            rerank_ms = (time.perf_counter() - t) * 1000
            reranked = True
    elif not want_rerank:
        reason = "not requested"

    # Caps AFTER reranking: they take from the top of the ordering, so applying
    # them to BM25 order would discard what the cross-encoder promotes.
    picked, per_ep, per_show = [], {}, {}
    for r in rows:
        stem, show = r[2], r[7]
        if max_per_ep and per_ep.get(stem, 0) >= max_per_ep:
            continue
        if max_per_show and per_show.get(show, 0) >= max_per_show:
            continue
        per_ep[stem] = per_ep.get(stem, 0) + 1
        per_show[show] = per_show.get(show, 0) + 1
        picked.append(r)
        if len(picked) >= k:
            break

    STATE["queries"] += 1
    if reranked:
        STATE["reranked"] += 1
    elif want_rerank:
        STATE["degraded"] += 1

    return {
        "query": query,
        "reranked": reranked,
        "rerank_reason": reason,
        "index": {
            "chunks": STATE["chunks"],
            "built": datetime.fromtimestamp(STATE["index_mtime"], timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "took_ms": {"fetch": round(fetch_ms), "rerank": round(rerank_ms),
                    "total": round(fetch_ms + rerank_ms)},
        "results": [{
            "rank": i,
            "show": r[7],
            "episode": r[1],
            "episode_number": r[9],
            "upload_date": r[3],
            "timestamp": f"{clock(r[4])}-{clock(r[5])}",
            "start_s": r[4],
            "text": r[0],
        } for i, r in enumerate(picked, 1)],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass                        # the access log is noise; we log what matters

    def do_GET(self):
        if self.path.split("?")[0] != "/health":
            return self._send(404, {"error": "not found"})
        try:
            open_index(STATE["index_path"]).close()
        except Exception as exc:
            return self._send(503, {"error": f"index unreadable: {exc}"})
        self._send(200, {
            "ok": True,
            "index": {"path": STATE["index_path"], "chunks": STATE["chunks"],
                      "built": datetime.fromtimestamp(STATE["index_mtime"], timezone.utc)
                                       .strftime("%Y-%m-%dT%H:%M:%SZ")},
            "shows": STATE["shows"],
            "model": STATE["model_name"],
            "device": STATE["device"],
            "model_loaded": STATE["model"] is not None,
            "loaded_at": STATE["loaded_at"],
            "counters": {"queries": STATE["queries"], "reranked": STATE["reranked"],
                         "degraded": STATE["degraded"]},
        })

    def do_POST(self):
        if self.path.split("?")[0] != "/search":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            return self._send(400, {"error": f"bad JSON: {exc}"})
        try:
            self._send(200, do_search(req))
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def keepwarm_loop(seconds):
    """See the docstring: 291 ms back-to-back against 563 ms with 20 s gaps."""
    while True:
        time.sleep(seconds)
        if STATE["model"] is None:
            continue
        try:
            with GPU_LOCK:
                STATE["model"].predict([("warm", "warm")], batch_size=1,
                                       show_progress_bar=False)
        except Exception as exc:
            log(f"keep-warm failed: {type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8422)
    ap.add_argument("--keepwarm", type=float, default=12.0, help="0 disables")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false",
                    help="serve BM25 only; no model is loaded at all")
    args = ap.parse_args()

    path = args.index
    if not os.path.exists(path):
        if os.path.exists(FALLBACK_INDEX):
            log(f"no local index at {path}, using {FALLBACK_INDEX} (~6x slower fetch)")
            path = FALLBACK_INDEX
        else:
            sys.exit(f"ABORT: no index at {path} -- run scripts/rebuild-index.sh")
    STATE["index_path"] = path
    open_index(path).close()

    if args.rerank:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _hf_env
        _hf_env.load()
        log(_hf_env.describe())
        from sentence_transformers import CrossEncoder
        device, kwargs = "cpu", {}
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                if args.fp16:
                    kwargs = {"model_kwargs": {"torch_dtype": torch.float16}}
        except ImportError:
            pass
        t = time.perf_counter()
        try:
            model = CrossEncoder(args.model, device=device, max_length=512, **kwargs)
        except Exception as exc:
            log(f"fp16 unavailable ({type(exc).__name__}), using fp32")
            model = CrossEncoder(args.model, device=device, max_length=512)
        # Burn the first inference here, not on a user's query: CUDA autotunes
        # on it and it measured ~2x a steady-state one.
        model.predict([("warm", "warm")], batch_size=1, show_progress_bar=False)
        STATE.update({"model": model, "model_name": args.model, "device": device,
                      "loaded_at": datetime.now(timezone.utc)
                                           .strftime("%Y-%m-%dT%H:%M:%SZ")})
        log(f"model {args.model} on {device}"
            f"{' (fp16)' if kwargs else ''} ready in {time.perf_counter() - t:.1f}s")
        if args.keepwarm:
            threading.Thread(target=keepwarm_loop, args=(args.keepwarm,),
                             daemon=True).start()
            log(f"keep-warm every {args.keepwarm}s")
    else:
        log("reranking disabled, serving BM25 order only")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"listening on http://{args.host}:{args.port}  "
        f"({STATE['chunks']:,} chunks, {len(STATE['shows'])} shows)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")


if __name__ == "__main__":
    main()
