"""PubMed corpus for continual pretraining.

Owner: Srinjoy. Source: ``MedRAG/pubmed`` (Xiong et al., 2024) - 23.9M title+abstract snippets
covering the PubMed articles that have both fields, pre-chunked into 1166 JSONL shards.

Using the MedRAG mirror rather than the NCBI baseline FTP trades some control for a great deal of
avoided work: the XML is already parsed, filtered to articles with usable abstracts, and chunked.

Size reality: the 1166 shards total roughly 70 GB, averaging ~60 MB each. That is why nothing here
downloads the whole corpus by default - callers take a shard budget, and every download is
preceded by a free-space check sized from the actual remote file sizes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

from tqdm.auto import tqdm

from medsel.data.base import BaseLoader
from medsel.data.contamination import fingerprint, resolve_exclusions
from medsel.registry import register_loader
from medsel.schema import CorpusDoc
from medsel.utils.diskguard import human_bytes, require_free

__all__ = ["PubMedCorpusLoader", "N_SHARDS", "shard_name"]

N_SHARDS = 1166
_SHARD_TEMPLATE = "chunk/pubmed23n{:04d}.jsonl"


def shard_name(index: int) -> str:
    """Remote filename of shard ``index`` (1-based, matching the upstream numbering)."""
    if not 1 <= index <= N_SHARDS:
        raise ValueError(f"shard index {index} out of range 1..{N_SHARDS}")
    return _SHARD_TEMPLATE.format(index)


@register_loader("pubmed")
class PubMedCorpusLoader(BaseLoader):
    """Shard-budgeted loader for the PubMed CPT corpus.

    Yields :class:`CorpusDoc` rather than ``QAExample`` - this is raw text for next-token
    prediction, not a QA task.

    Each raw row carries three text fields: ``title``, ``content`` (the abstract), and
    ``contents``, which is exactly ``title + " " + content`` pre-joined for retrieval. Reading
    ``contents`` alongside the other two would roughly double memory and disk for no new
    information, so only ``title`` and ``content`` are kept and the join is left to
    :attr:`CorpusDoc.full_text`.
    """

    hf_id = "MedRAG/pubmed"
    splits: ClassVar[tuple[str, ...]] = ("train",)
    record_type: ClassVar[type] = CorpusDoc
    normalizer_version: ClassVar[int] = 1

    def __init__(
        self,
        num_shards: int = 8,
        shard_offset: int = 0,
        min_chars: int = 200,
        dedup: bool = True,
        shard_dir: str | Path | None = None,
        exclude_pmids: str | Path | Iterable[int | str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            num_shards: How many shards to read. Each is ~60 MB and ~20k documents.
            shard_offset: 0-based offset of the first shard, so collaborators can split the
                corpus between them without overlap.
            min_chars: Drop documents whose abstract is shorter than this. PubMed contains many
                title-only records that contribute little to CPT and skew packing.
            dedup: Drop repeat PMIDs. Holds one int per seen document in memory.
            shard_dir: Read shards from this local directory instead of downloading, for shared
                lab machines where one person has already fetched the corpus.
            exclude_pmids: Documents to keep out of the corpus. Pass ``"pubmedqa"`` before any
                run whose model will later be scored on PubMedQA, since that benchmark is built
                from these same abstracts. Also accepts a path to a JSON list of ints or an
                iterable of ints. See :mod:`medsel.data.contamination`.
        """
        super().__init__(**kwargs)
        if not 1 <= num_shards <= N_SHARDS:
            raise ValueError(f"num_shards must be in 1..{N_SHARDS}, got {num_shards}")
        if not 0 <= shard_offset < N_SHARDS:
            raise ValueError(f"shard_offset must be in 0..{N_SHARDS - 1}, got {shard_offset}")

        self.num_shards = num_shards
        self.shard_offset = shard_offset
        self.min_chars = min_chars
        self.dedup = dedup
        self.shard_dir = Path(shard_dir) if shard_dir else None
        self.exclude_pmids = resolve_exclusions(exclude_pmids)

    @property
    def cache_key(self) -> str:
        # The exclusion fingerprint is part of the key on purpose: a parquet built without the
        # contamination filter must never be served to a run that asked for it.
        return (
            f"{self.name}-s{self.shard_offset}+{self.num_shards}"
            f"-min{self.min_chars}-dedup{int(self.dedup)}"
            f"-excl{fingerprint(self.exclude_pmids)}-v{self.normalizer_version}"
        )

    def shard_names(self) -> list[str]:
        """Remote filenames this loader will read, clipped at the end of the corpus."""
        last = min(self.shard_offset + self.num_shards, N_SHARDS)
        return [shard_name(i) for i in range(self.shard_offset + 1, last + 1)]

    def remote_size_bytes(self) -> int:
        """Total download size of the selected shards, read from the Hub rather than guessed."""
        from huggingface_hub import HfApi

        wanted = set(self.shard_names())
        info = HfApi().repo_info(self.hf_id, repo_type="dataset", files_metadata=True)
        return sum(s.size or 0 for s in (info.siblings or []) if s.rfilename in wanted)

    def download_shards(self, progress: bool = True) -> list[Path]:
        """Fetch the selected shards into the HF cache, refusing to start if disk is short."""
        from huggingface_hub import hf_hub_download

        names = self.shard_names()
        if self.shard_dir is not None:
            paths = [self.shard_dir / Path(n).name for n in names]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} shard(s) absent from {self.shard_dir}, "
                    f"first missing: {missing[0]}"
                )
            return paths

        needed = self.remote_size_bytes()
        require_free(needed, Path.home())

        paths = []
        with tqdm(names, desc=f"download {self.name}", unit="shard", disable=not progress) as bar:
            for name in bar:
                paths.append(
                    Path(
                        hf_hub_download(
                            self.hf_id, filename=name, repo_type="dataset", revision=self.revision
                        )
                    )
                )
        return paths

    def raw(self, split: str = "train", streaming: bool = True) -> Iterator[dict[str, Any]]:
        """Yield raw JSON rows from the selected shards. Always streams, line by line."""
        self.check_split(split)
        for path in self.download_shards():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    def normalize(self, row: dict[str, Any], idx: int, split: str) -> CorpusDoc:
        pmid = row.get("PMID")
        return CorpusDoc(
            uid=f"{self.name}/{row.get('id') or pmid or idx}",
            source=self.name,
            title=str(row.get("title") or "").strip(),
            # 'contents' is title + " " + content upstream; recomputed by full_text, so ignored.
            text=str(row.get("content") or "").strip(),
            meta={"PMID": pmid},
        )

    def load(
        self,
        split: str = "train",
        limit: int | None = None,
        streaming: bool = True,
        progress: bool = True,
    ) -> Iterator[CorpusDoc]:
        """Yield filtered, deduplicated documents from the shard budget."""
        seen: set[int] = set()
        kept = 0
        skipped_short = 0
        skipped_dupe = 0
        skipped_contam = 0

        with tqdm(
            total=limit, desc=f"{self.name}:{split}", unit="doc", disable=not progress, leave=False
        ) as bar:
            for idx, row in enumerate(self.raw(split)):
                if limit is not None and kept >= limit:
                    break

                pmid = row.get("PMID")

                # Contamination first: a document that must not be trained on should not be
                # counted as a duplicate or a short abstract either.
                if self.exclude_pmids and isinstance(pmid, int) and pmid in self.exclude_pmids:
                    skipped_contam += 1
                    continue

                if self.dedup and isinstance(pmid, int):
                    if pmid in seen:
                        skipped_dupe += 1
                        continue
                    seen.add(pmid)

                text = str(row.get("content") or "").strip()
                if len(text) < self.min_chars:
                    skipped_short += 1
                    continue

                yield self.normalize(row, idx, split)
                kept += 1
                bar.update(1)

            bar.set_postfix(short=skipped_short, dupe=skipped_dupe, contam=skipped_contam)

    def stats(
        self, split: str = "train", limit: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        n = 0
        total_chars = 0
        with_title = 0
        for doc in self.load(split, limit=limit, progress=kwargs.get("progress", True)):
            n += 1
            total_chars += len(doc.full_text)
            with_title += bool(doc.title)
        return {
            "source": self.name,
            "split": split,
            "hf_id": self.hf_id,
            "shards": f"{self.shard_offset + 1}..{self.shard_offset + self.num_shards}",
            "n_excluded_pmids": len(self.exclude_pmids),
            "n_documents": n,
            "n_with_title": with_title,
            "mean_chars": round(total_chars / n, 1) if n else 0.0,
            "total_chars": total_chars,
            "approx_size": human_bytes(total_chars),
        }
