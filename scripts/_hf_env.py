#!/usr/bin/env python3
"""
_hf_env.py -- make HF_TOKEN available to the rerank scripts.

WHY. sentence-transformers checks the HuggingFace Hub for model updates on every
CrossEncoder load. Unauthenticated, that prints a rate-limit warning and gets the
slow anonymous tier:

    Warning: You are sending unauthenticated requests to the HF Hub.
    Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Harmless, but it also means a Hub hiccup or a rate limit can stall a model load
that should be instant from cache.

RESOLUTION ORDER, first hit wins:
  1. HF_TOKEN already in the environment -- an explicit export always beats a file
  2. ~/.config/scriberr/hf.env
  3. /etc/scriberr-feed.env

PERMISSIONS, AND WHY THERE ARE TWO CANDIDATE PATHS. /etc/scriberr-feed.env is
root-owned mode 600 because it holds the Scriberr API key and systemd reads it as
root. The rerank scripts run as an ordinary user, WHICH CANNOT READ THAT FILE.
So either:
  (a) put HF_TOKEN in ~/.config/scriberr/hf.env, mode 600, owned by the user --
      recommended, since only user-run CLI tools need it and the API key stays
      root-only; or
  (b) loosen /etc/scriberr-feed.env to 640 root:nas, which also exposes the
      Scriberr API key to everyone in that group.

Unreadable candidates are skipped silently -- a missing file is the normal case,
not an error. Parsing is deliberately dumb: KEY=VALUE, '#' comments, optional
surrounding quotes. It is not a shell parser and does not try to be.

THE TOKEN VALUE IS NEVER PRINTED OR LOGGED. `describe()` reports only which
source supplied it.
"""

import os

CANDIDATES = [
    os.path.expanduser("~/.config/scriberr/hf.env"),
    "/etc/scriberr-feed.env",
]

# huggingface_hub has read both names across versions; set both so behaviour
# does not depend on which one the installed version happens to check.
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

_source = None


def _read(path, key="HF_TOKEN"):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() != key:
                    continue
                v = v.strip().strip('"').strip("'")
                return v or None
    except (OSError, PermissionError):
        return None                     # missing or root-only: both normal
    return None


def load(extra_paths=()):
    """Populate HF_TOKEN in os.environ if not already set. Returns True if a
    token is available by the time this returns."""
    global _source
    for var in TOKEN_VARS:
        if os.environ.get(var):
            _source = f"environment ({var})"
            for other in TOKEN_VARS:
                os.environ.setdefault(other, os.environ[var])
            return True

    for path in list(extra_paths) + CANDIDATES:
        token = _read(path)
        if token:
            for var in TOKEN_VARS:
                os.environ[var] = token
            _source = path
            return True

    _source = None
    return False


def describe():
    """One line for the log. NEVER includes the token itself."""
    if _source:
        return f"HF token from {_source}"
    return ("no HF token found (unauthenticated Hub access; expect a rate-limit "
            "warning). See scripts/_hf_env.py for where to put one.")
