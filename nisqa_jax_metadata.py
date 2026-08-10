"""Stdlib-only metadata integrity primitives shared by loader and release tools."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

_CANONICAL_HASH_EXCLUDED_FIELDS: frozenset[str] = frozenset({"npz_sha256", "metadata_sha256"})


def canonical_metadata_checksum(metadata: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 of semantic artifact metadata.

    File-reference hashes are excluded because ``npz_sha256`` pins the weight
    bytes and ``metadata_sha256`` is this checksum itself. JSON formatting does
    not affect the result because the semantic payload is serialized with sorted
    keys and compact separators.
    """
    payload = {key: value for key, value in metadata.items() if key not in _CANONICAL_HASH_EXCLUDED_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
