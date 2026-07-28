"""PubMed to PubMedQA contamination control.

PubMedQA is built from PubMed abstracts and our CPT corpus is PubMed abstracts. Continually
pretraining on PubMed and then evaluating on PubMedQA can mean scoring the model on text it was
just trained on. That does not crash and it does not look wrong; it produces a plausible number
that is not a measurement of anything. This module builds the PMID exclusion list that makes the
training corpus and the evaluation set disjoint.

Both sides key on an integer PMID: PubMedQA calls it ``pubid``, the ``MedRAG/pubmed`` corpus calls
it ``PMID``. Verified against the actual data rather than the dataset cards.

Typical use is through the loader rather than directly::

    get_loader("pubmed", num_shards=32, exclude_pmids="pubmedqa")

which resolves the string through :func:`resolve_exclusions` below.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "DEFAULT_CONFIGS",
    "EVAL_CONFIGS",
    "fingerprint",
    "pubmedqa_pmids",
    "resolve_exclusions",
]

# Configs whose PMIDs must be excluded from any CPT corpus used before a PubMedQA evaluation.
# ``pqa_labeled`` is the evaluation set, so it is the minimum. ``pqa_artificial`` matters only if
# it is also used for SFT, in which case excluding it keeps the CPT and SFT corpora disjoint too.
DEFAULT_CONFIGS: tuple[str, ...] = ("pqa_labeled",)
EVAL_CONFIGS: tuple[str, ...] = ("pqa_labeled", "pqa_artificial", "pqa_unlabeled")

_HF_ID = "qiaojin/PubMedQA"
_CACHE_DIR = Path("data/contamination")


def fingerprint(pmids: Iterable[int]) -> str:
    """Short stable digest of an exclusion set, for cache keys.

    Two different exclusion sets must produce different cache keys, or a parquet built without
    the filter could be served to a run that asked for it. Sorted so set iteration order cannot
    change the answer.
    """
    ids = sorted(set(pmids))
    if not ids:
        return "none"
    digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:10]
    return f"{len(ids)}@{digest}"


def _cache_path(configs: tuple[str, ...], cache_dir: Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    return root / f"pubmedqa_pmids_{'+'.join(sorted(configs))}.json"


def pubmedqa_pmids(
    configs: Iterable[str] = DEFAULT_CONFIGS,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    progress: bool = True,
) -> frozenset[int]:
    """Collect the PubMedQA PMIDs to keep out of a CPT corpus.

    Cached to disk after the first build, because this is needed by every CPT run and the answer
    only changes when the upstream dataset does.

    Args:
        configs: PubMedQA configs to pull PMIDs from. Defaults to the evaluation set alone.
        cache_dir: Where to store the built list. Defaults to ``data/contamination/``.
        refresh: Rebuild from the Hub even if a cached list exists.
        progress: Show a progress bar while reading configs.

    Returns:
        The PMIDs as a frozenset of ints.
    """
    keys = tuple(sorted(configs))
    if not keys:
        raise ValueError("configs is empty; pass at least one PubMedQA config")

    path = _cache_path(keys, Path(cache_dir) if cache_dir is not None else None)
    if path.exists() and not refresh:
        return frozenset(json.loads(path.read_text()))

    from datasets import load_dataset
    from tqdm.auto import tqdm

    pmids: set[int] = set()
    with tqdm(keys, desc="pubmedqa pmids", unit="config", disable=not progress) as bar:
        for config in bar:
            rows = load_dataset(_HF_ID, config, split="train")
            pmids.update(int(p) for p in rows["pubid"] if p is not None)
            bar.set_postfix(pmids=len(pmids))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(pmids)))
    return frozenset(pmids)


def resolve_exclusions(
    spec: str | Path | Iterable[int | str] | None, progress: bool = True
) -> frozenset[int]:
    """Turn a user-supplied exclusion spec into a concrete PMID set.

    Accepts the shapes a config file or a caller might reasonably supply:

    - ``None`` -> empty set, no filtering
    - ``"pubmedqa"`` -> the default PubMedQA evaluation PMIDs, fetched and cached
    - ``"pubmedqa:pqa_labeled+pqa_artificial"`` -> those specific configs
    - a path to a JSON file holding a list of ints
    - any iterable of ints
    """
    if spec is None:
        return frozenset()

    if isinstance(spec, str):
        if spec.startswith("pubmedqa"):
            _, _, rest = spec.partition(":")
            configs = tuple(rest.split("+")) if rest else DEFAULT_CONFIGS
            unknown = set(configs) - set(EVAL_CONFIGS)
            if unknown:
                raise ValueError(
                    f"unknown PubMedQA config(s): {', '.join(sorted(unknown))}; "
                    f"available: {', '.join(EVAL_CONFIGS)}"
                )
            return pubmedqa_pmids(configs, progress=progress)
        spec = Path(spec)

    if isinstance(spec, Path):
        if not spec.exists():
            raise FileNotFoundError(f"exclusion list not found: {spec}")
        return frozenset(int(p) for p in json.loads(spec.read_text()))

    return frozenset(int(p) for p in spec)
