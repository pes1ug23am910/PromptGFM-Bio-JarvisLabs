"""
Convert Markdown files to PDF while preserving table layout and formatting.

Default usage targets data/README.md and writes data/README.pdf.

Examples:
    python scripts/md_to_pdf.py
    python scripts/md_to_pdf.py --input data/README.md --output data/README.pdf
    python scripts/md_to_pdf.py --input docs/guide.md --css docs/pdf_style.css
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


DEFAULT_CSS = """
@page {
    size: A4;
    margin: 20mm 16mm;
}

body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1f2937;
}

h1, h2, h3, h4 {
    color: #111827;
    margin-top: 1.1em;
    margin-bottom: 0.45em;
    page-break-after: avoid;
}

h1 {
    font-size: 24pt;
    border-bottom: 1px solid #d1d5db;
    padding-bottom: 8px;
}

h2 {
    font-size: 18pt;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
}

h3 {
    font-size: 14pt;
}

p, ul, ol, blockquote {
    margin-top: 0.35em;
    margin-bottom: 0.55em;
}

code {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9.8pt;
    background: #f3f4f6;
    border-radius: 4px;
    padding: 1px 4px;
}

pre {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9pt;
    background: #111827;
    color: #f9fafb;
    border-radius: 6px;
    padding: 10px 12px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

pre code {
    background: transparent;
    color: inherit;
    padding: 0;
}

/* Keep Markdown table structure clearly rendered in PDF */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 14px;
    font-size: 10pt;
    table-layout: auto;
}

thead {
    display: table-header-group;
}

tbody tr {
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #cbd5e1;
    padding: 7px 9px;
    vertical-align: top;
    word-break: break-word;
}

th {
    background: #e2e8f0;
    font-weight: 700;
    text-align: left;
}

tbody tr:nth-child(even) {
    background: #f8fafc;
}

blockquote {
    border-left: 4px solid #cbd5e1;
    padding-left: 10px;
    color: #4b5563;
}

a {
    color: #0f766e;
    text-decoration: none;
}

hr {
    border: none;
    border-top: 1px solid #d1d5db;
    margin: 12px 0;
}
"""


def parse_args() -> argparse.Namespace:
    default_backend = "xhtml2pdf" if sys.platform.startswith("win") else "auto"

    parser = argparse.ArgumentParser(
        description="Convert Markdown to PDF with table-preserving formatting."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/README.md",
        help="Path to input Markdown file (default: data/README.md)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output PDF file (default: same name as input with .pdf)",
    )
    parser.add_argument(
        "--css",
        default=None,
        help="Optional custom CSS file to append after default styling.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional HTML title for the generated document.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "weasyprint", "xhtml2pdf"],
        default=default_backend,
        help=f"PDF backend to use (default: {default_backend}).",
    )
    return parser.parse_args()


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_markdown_module() -> object:
    try:
        import markdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'markdown'. Install with: pip install markdown"
        ) from exc

    return markdown


def load_pdf_backend(preferred_backend: str) -> tuple[str, object]:
    weasy_error = None

    if preferred_backend in ("auto", "weasyprint"):
        try:
            from weasyprint import HTML  # type: ignore

            return "weasyprint", HTML
        except Exception as exc:
            weasy_error = exc
            if preferred_backend == "weasyprint":
                raise RuntimeError(
                    "WeasyPrint is unavailable in this environment. "
                    "Either install its system libraries, or use --backend xhtml2pdf. "
                    f"Original error: {exc}"
                ) from exc

    try:
        from xhtml2pdf import pisa  # type: ignore

        return "xhtml2pdf", pisa
    except ImportError as exc:
        install_msg = (
            "No usable PDF backend found. Install one of:\n"
            "  pip install weasyprint\n"
            "  pip install xhtml2pdf\n"
            "For this Windows env, xhtml2pdf is recommended."
        )
        if weasy_error is not None:
            install_msg += f"\nWeasyPrint error: {weasy_error}"
        raise RuntimeError(
            install_msg
        ) from exc


def markdown_to_html(markdown_text: str, title: str, css_text: str, markdown_module: object) -> str:
    rendered_body = markdown_module.markdown(  # type: ignore[attr-defined]
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "nl2br",
        ],
        output_format="html5",
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>{css_text}</style>
</head>
<body>
    <main class=\"markdown-body\">
        {rendered_body}
    </main>
</body>
</html>
"""


def convert(
    md_path: Path,
    pdf_path: Path,
    custom_css_path: Path | None = None,
    title: str | None = None,
    backend_preference: str = "auto",
) -> str:
    markdown_module = load_markdown_module()
    backend_name, pdf_backend = load_pdf_backend(backend_preference)

    markdown_text = md_path.read_text(encoding="utf-8")
    combined_css = DEFAULT_CSS

    if custom_css_path is not None:
        custom_css = custom_css_path.read_text(encoding="utf-8")
        combined_css = f"{DEFAULT_CSS}\n\n{custom_css}"

    resolved_title = title or md_path.stem
    html_document = markdown_to_html(markdown_text, resolved_title, combined_css, markdown_module)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if backend_name == "weasyprint":
        pdf_backend(string=html_document, base_url=str(md_path.parent.resolve())).write_pdf(str(pdf_path))
    else:
        with pdf_path.open("wb") as output_file:
            pisa_status = pdf_backend.CreatePDF(src=html_document, dest=output_file, encoding="utf-8")
        if getattr(pisa_status, "err", 1):
            raise RuntimeError("xhtml2pdf failed while rendering the PDF.")

    return backend_name


def main() -> int:
    args = parse_args()

    md_path = resolve_path(args.input, PROJECT_ROOT)
    if not md_path.exists():
        print(f"Input file not found: {md_path}")
        print("Tip: relative paths are resolved from the project root.")
        return 1

    if args.output:
        pdf_path = resolve_path(args.output, PROJECT_ROOT)
    else:
        pdf_path = md_path.with_suffix(".pdf")

    custom_css_path = None
    if args.css:
        custom_css_path = resolve_path(args.css, PROJECT_ROOT)
        if not custom_css_path.exists():
            print(f"Custom CSS file not found: {custom_css_path}")
            return 1

    try:
        used_backend = convert(
            md_path,
            pdf_path,
            custom_css_path=custom_css_path,
            title=args.title,
            backend_preference=args.backend,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"Conversion failed: {exc}")
        return 1

    print(f"PDF created: {pdf_path} (backend: {used_backend})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
