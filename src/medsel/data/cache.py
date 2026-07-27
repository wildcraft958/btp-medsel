"""Parquet cache for normalised records.

Four people re-deriving the same subsets from the Hub is slow and, worse, silently divergent when
a loader changes. Every cache entry carries a manifest pinning the loader's cache key, the record
dataclass' field signature, and the row count. Any drift invalidates the entry and forces a
rebuild, so a stale parquet can never masquerade as current data.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from medsel import __version__
from medsel.schema import CorpusDoc, QAExample
from medsel.utils.diskguard import free_bytes, human_bytes, require_free

__all__ = [
    "cache_root",
    "cache_paths",
    "write_cache",
    "read_cache",
    "read_manifest",
    "cache_is_valid",
    "clear_cache",
]

# Fields held as dicts in Python but stored as JSON text, so the parquet schema stays fixed
# regardless of which keys a particular dataset happens to populate.
_JSON_FIELDS = frozenset({"options", "labels", "meta"})

_BATCH_ROWS = 1000
_MIN_FREE_BYTES = 256 * 1024 * 1024

_RECORD_TYPES: dict[str, type] = {"QAExample": QAExample, "CorpusDoc": CorpusDoc}


def cache_root() -> Path:
    """Cache location. Override with ``MEDSEL_CACHE`` to use a larger volume."""
    return Path(os.environ.get("MEDSEL_CACHE", "data/cache"))


def cache_paths(source: str, split: str, root: Path | None = None) -> tuple[Path, Path]:
    base = (root or cache_root()) / source
    return base / f"{split}.parquet", base / f"{split}.manifest.json"


def _schema_hash(record_type: type) -> str:
    signature = ",".join(f.name for f in dataclass_fields(record_type))
    return hashlib.sha256(signature.encode()).hexdigest()[:16]


def _encode(record: QAExample | CorpusDoc) -> dict[str, Any]:
    row = record.to_dict()
    for key in _JSON_FIELDS & row.keys():
        row[key] = None if row[key] is None else json.dumps(row[key], ensure_ascii=False)
    return row


def _flush(
    batch: list[dict[str, Any]], writer: pq.ParquetWriter | None, path: Path
) -> pq.ParquetWriter:
    """Append one batch, opening the writer on first use. Returns the live writer."""
    table = pa.Table.from_pylist(batch)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)

    available = free_bytes(path.parent)
    if available < _MIN_FREE_BYTES:
        raise OSError(
            f"only {human_bytes(available)} left while writing {path}; "
            f"aborting before the filesystem fills"
        )
    return writer


def _decode(row: dict[str, Any], record_type: type) -> QAExample | CorpusDoc:
    data = dict(row)
    for key in _JSON_FIELDS & data.keys():
        if isinstance(data[key], str):
            data[key] = json.loads(data[key])
    if "contexts" in data and data["contexts"] is None:
        data["contexts"] = []
    return record_type.from_dict(data)


def write_cache(
    records: Iterable[QAExample | CorpusDoc],
    source: str,
    split: str,
    cache_key: str,
    record_type: type = QAExample,
    root: Path | None = None,
    extra: dict[str, Any] | None = None,
    progress: bool = True,
) -> Path:
    """Stream ``records`` to parquet and write the accompanying manifest.

    Rows are flushed in batches so corpora larger than memory stay writable, and free space is
    re-checked at every flush - a long PubMed ingest must fail early, not fill the disk.
    """
    parquet_path, manifest_path = cache_paths(source, split, root)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    require_free(_MIN_FREE_BYTES, parquet_path.parent)

    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    n_rows = 0

    try:
        with tqdm(desc=f"cache {source}:{split}", unit="row", disable=not progress) as bar:
            for record in records:
                batch.append(_encode(record))
                n_rows += 1
                bar.update(1)
                if len(batch) >= _BATCH_ROWS:
                    writer = _flush(batch, writer, parquet_path)
                    batch.clear()
            if batch:
                writer = _flush(batch, writer, parquet_path)
                batch.clear()
    except BaseException:
        if writer is not None:
            writer.close()
        parquet_path.unlink(missing_ok=True)
        raise

    if writer is None:  # no rows at all - still emit an empty, valid table
        pq.write_table(pa.Table.from_pylist([]), parquet_path)
    else:
        writer.close()

    manifest = {
        "source": source,
        "split": split,
        "cache_key": cache_key,
        "record_type": record_type.__name__,
        "schema_hash": _schema_hash(record_type),
        "n_rows": n_rows,
        "medsel_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return parquet_path


def read_manifest(source: str, split: str, root: Path | None = None) -> dict[str, Any] | None:
    _, manifest_path = cache_paths(source, split, root)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def cache_is_valid(
    source: str,
    split: str,
    cache_key: str,
    record_type: type = QAExample,
    root: Path | None = None,
) -> bool:
    """True only if the cached entry was produced by this exact loader and record schema."""
    parquet_path, _ = cache_paths(source, split, root)
    manifest = read_manifest(source, split, root)
    if manifest is None or not parquet_path.exists():
        return False
    return (
        manifest.get("cache_key") == cache_key
        and manifest.get("record_type") == record_type.__name__
        and manifest.get("schema_hash") == _schema_hash(record_type)
    )


def read_cache(
    source: str,
    split: str,
    root: Path | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> Iterator[QAExample | CorpusDoc]:
    """Yield decoded records from a cached parquet file."""
    parquet_path, _ = cache_paths(source, split, root)
    if not parquet_path.exists():
        raise FileNotFoundError(f"no cache at {parquet_path}; run `medsel data prepare` first")

    manifest = read_manifest(source, split, root) or {}
    type_name = str(manifest.get("record_type", "QAExample"))
    record_type: type = _RECORD_TYPES[type_name] if type_name in _RECORD_TYPES else QAExample

    parquet_file = pq.ParquetFile(parquet_path)
    total = min(limit, parquet_file.metadata.num_rows) if limit else parquet_file.metadata.num_rows

    emitted = 0
    with tqdm(
        total=total, desc=f"read {source}:{split}", unit="row", disable=not progress, leave=False
    ) as bar:
        for batch in parquet_file.iter_batches(batch_size=_BATCH_ROWS):
            for row in batch.to_pylist():
                if limit is not None and emitted >= limit:
                    return
                yield _decode(row, record_type)
                emitted += 1
                bar.update(1)


def clear_cache(source: str | None = None, root: Path | None = None) -> int:
    """Delete cached entries. Returns the number of files removed."""
    base = root or cache_root()
    target = base / source if source else base
    if not target.exists():
        return 0
    removed = 0
    for path in sorted(target.rglob("*")):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed
