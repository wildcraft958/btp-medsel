"""Prompt templates.

One renderer, used by both evaluation and SFT. If the two drifted apart, every "does CPT help
downstream?" comparison would silently confound a data effect with a formatting effect.

:data:`TEMPLATE_VERSION` is recorded in every evaluation report. Numbers produced under different
template versions are not comparable, and this makes that visible instead of leaving it implicit.
"""

from __future__ import annotations

from medsel.schema import QAExample

__all__ = ["TEMPLATE_VERSION", "render_prompt", "render_continuation", "ScoringMode"]

TEMPLATE_VERSION = 1

ScoringMode = str  # "letter" | "text"

_DEFAULT_MAX_CONTEXT_CHARS = 2000


def _render_contexts(example: QAExample, max_chars: int) -> str:
    if not example.contexts:
        return ""
    joined = "\n".join(example.contexts).strip()
    if max_chars and len(joined) > max_chars:
        # Truncate the head, not the tail: PubMedQA abstracts put conclusions last, and the
        # question is usually answerable from the later sections.
        joined = "..." + joined[-max_chars:]
    return joined


def render_prompt(
    example: QAExample,
    include_contexts: bool = True,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Render an example into a prompt ending at ``Answer:``.

    The trailing space belongs to the continuation, not the prompt, so that tokenizers which
    attach leading whitespace to a word produce the same token either way.
    """
    if not example.options:
        raise ValueError(f"{example.uid} has no options to render")

    parts = []
    if include_contexts:
        context = _render_contexts(example, max_context_chars)
        if context:
            parts.append(f"Context: {context}")

    parts.append(f"Question: {example.question}")
    parts.extend(f"{key}. {text}" for key, text in sorted(example.options.items()))
    parts.append("Answer:")
    return "\n".join(parts)


def render_continuation(example: QAExample, key: str, mode: ScoringMode = "letter") -> str:
    """Render the scored continuation for one option.

    ``letter`` scores " A", the convention published MCQ numbers use. ``text`` scores the option
    wording instead, which base models handle far better - a model that has never been
    instruction-tuned has no reason to know that "A" is a valid thing to say. Since this project
    evaluates base models before and after CPT, both modes matter.
    """
    if not example.options or key not in example.options:
        raise ValueError(f"{example.uid} has no option {key!r}")

    if mode == "letter":
        return f" {key}"
    if mode == "text":
        return f" {example.options[key]}"
    raise ValueError(f"unknown scoring mode {mode!r}; use 'letter' or 'text'")
