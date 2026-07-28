"""Prompt rendering, shared by evaluation and (later) SFT."""

from medsel.prompts.templates import (
    TEMPLATE_VERSION,
    render_continuation,
    render_prompt,
)

__all__ = ["TEMPLATE_VERSION", "render_prompt", "render_continuation"]
