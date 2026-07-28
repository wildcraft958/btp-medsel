"""Build the shareable PDF: literature review plus the July 2026 research findings appendix.

Produces a single document a collaborator can read without cloning the repo. The appendix is the
raw research output, included so the evidence behind the 2026 additions can be checked rather than
taken on trust.

Usage::

    uv pip install markdown
    uv run python scripts/build_pdf.py

Markdown to HTML via ``markdown``, HTML to PDF via headless Chrome. Chrome rather than weasyprint
because the latter needs a working system install that is easy to break, whereas any machine with
a browser can run this.

No em or en dashes are introduced anywhere, matching the repo rule enforced by
``tests/test_style.py``, and :func:`strip_dashes` is a belt-and-braces guard on the way out.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
FINDINGS = ROOT / "docs" / "research_findings_2026-07.json"
BUILT = "2026-07-29"

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; page-break-after: avoid; }
h2 { font-size: 14pt; margin: 18pt 0 6pt 0; padding-bottom: 3pt;
     border-bottom: 1px solid #bbb; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 13pt 0 4pt 0; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt 0; page-break-after: avoid; }
p { margin: 0 0 6pt 0; text-align: justify; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt;
       background: #f4f4f4; padding: 0 2px; border-radius: 2px; }
pre { background: #f6f6f6; border-left: 3px solid #ccc; padding: 6pt 8pt;
      font-size: 8pt; overflow-wrap: break-word; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eee; font-weight: bold; }
blockquote { border-left: 3px solid #ccc; margin: 6pt 0; padding-left: 10pt; color: #444; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin-bottom: 2pt; }
a { color: #14507a; text-decoration: none; }
.title-block { text-align: center; margin-bottom: 16pt; padding-bottom: 10pt;
               border-bottom: 2px solid #333; }
.title-block .sub { font-size: 11pt; color: #444; margin-top: 4pt; }
.title-block .meta { font-size: 9pt; color: #666; margin-top: 8pt; }
.verdict { font-weight: bold; }
.killed { color: #8a1c1c; }
.survived { color: #14561f; }
.pagebreak { page-break-before: always; }
.claim { page-break-inside: avoid; margin-bottom: 9pt; padding-bottom: 7pt;
         border-bottom: 1px dotted #ccc; }
.claim .src { font-size: 8pt; color: #555; }
.claim .quote { font-size: 8.5pt; color: #333; font-style: italic;
                border-left: 2px solid #ddd; padding-left: 7pt; margin-top: 3pt; }
"""


def strip_dashes(text: str) -> str:
    """Guard: the source is dash-free, but never let one reach a shared document.

    Codepoints rather than literals so this file does not trip the check in
    ``tests/test_style.py`` that it exists to support.
    """
    return text.replace(chr(0x2014), ", ").replace(chr(0x2013), "-")


def build_appendix() -> str:
    data = json.load(FINDINGS.open())
    parts = [
        '<div class="pagebreak"></div>',
        "<h1>Appendix: Deep Research Findings, July 2026</h1>",
        "<p>This appendix records the raw output of the research pass that produced the 2026 "
        "additions to the review above. It is included so the evidence can be checked rather "
        "than taken on trust.</p>",
        "<p><b>Method.</b> Five parallel search agents covered corpus-scale selection, the "
        "multi-stage and multi-target novelty check, follow-ups to the contested premise that "
        "medical adaptation helps, capability retention and non-MCQ benchmarks, and an open "
        "sweep. Twenty-five sources were read and 125 candidate claims extracted. Each claim "
        "that reached verification was then given to three independent agents instructed to "
        "<i>refute</i> it, with a majority refutation killing the claim. Sixteen of twenty-five "
        "survived.</p>",
        "<p><b>How to read this.</b> Surviving claims are evidence, not proof: they survived an "
        "attempt to break them. Killed claims are listed too, because a reader who finds the same "
        "paper independently should know the claim was tested and did not hold. Where a figure "
        "came from an abstract rather than a paper body, the claim says so.</p>",
        f'<h2><span class="survived">Surviving claims ({len(data["survived"])})</span></h2>',
    ]
    for i, r in enumerate(data["survived"], 1):
        parts.append(
            f'<div class="claim"><p><b>S{i}.</b> {strip_dashes(r["claim"])}</p>'
            f'<p class="src">Source: <a href="{r["url"]}">{r["url"]}</a> ({r["tier"]}) '
            f"&middot; refuted by {r['n_refuted']} of {r['n_votes']} verifiers</p>"
            + (f'<p class="quote">{strip_dashes(r["quote"])}</p>' if r["quote"] else "")
            + "</div>"
        )
    parts.append(
        f'<h2><span class="killed">Claims killed by verification '
        f"({len(data['killed'])})</span></h2>"
        "<p>These did not survive. They are recorded so nobody re-derives them.</p>"
    )
    for i, r in enumerate(data["killed"], 1):
        parts.append(
            f'<div class="claim"><p><b>K{i}.</b> {strip_dashes(r["claim"])}</p>'
            f'<p class="src">Source: <a href="{r["url"]}">{r["url"]}</a> ({r["tier"]}) '
            f'&middot; <span class="killed">refuted by {r["n_refuted"]} of {r["n_votes"]} '
            f"verifiers</span></p></div>"
        )
    return "\n".join(parts)


def main() -> int:
    review = (ROOT / "docs" / "literature_review.md").read_text(encoding="utf-8")

    # The in-document contents list points at HTML anchors that do not survive PDF conversion
    # cleanly, and a PDF has its own page numbers. Drop it rather than ship dead links.
    review = re.sub(r"\n## Contents\n.*?\n---\n", "\n", review, count=1, flags=re.S)
    # The H1 is replaced by the styled title block below.
    review = re.sub(r"^# .*?\n", "", review, count=1)

    body = markdown.markdown(
        strip_dashes(review),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )

    title = (
        '<div class="title-block">'
        "<h1>Multi-Stage and Multi-Target Data Selection for Medical Foundation Models</h1>"
        '<div class="sub">Literature Review and 2026 Research Findings</div>'
        '<div class="meta">B.Tech Project &middot; Animesh Raj, Debmalya, Arkajyoti, Srinjoy'
        f"<br>Built {BUILT} from the medsel repository</div></div>"
    )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Multi-Stage Medical Data Selection: Literature Review</title>"
        f"<style>{CSS}</style></head><body>{title}{body}{build_appendix()}</body></html>"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / ".review.tmp.html"
    html_path.write_text(html, encoding="utf-8")

    # Chrome rather than weasyprint: the local weasyprint install has a broken tinycss2
    # dependency, and Chrome renders the tables and page breaks correctly anyway.
    pdf_path = OUT_DIR / "medsel_literature_review_2026-07.pdf"
    try:
        result = subprocess.run(
            [
                "google-chrome",
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--virtual-time-budget=20000",
                # Otherwise Chrome stamps a browser print header with today's date and the tab
                # title on every page, which reads as an accident in a shared document.
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                f"file://{html_path}",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError:
        print(
            "google-chrome not found. Install Chrome or Chromium, or point this script at "
            "another HTML-to-PDF renderer.",
            file=sys.stderr,
        )
        return 1
    finally:
        # The intermediate HTML is a build artefact, not a deliverable. Removed even on failure
        # so a broken run does not leave an untracked file next to the committed PDF.
        html_path.unlink(missing_ok=True)

    if result.returncode != 0 or not pdf_path.exists():
        print("chrome failed:", result.stderr[-1500:], file=sys.stderr)
        return 1

    print(f"wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
