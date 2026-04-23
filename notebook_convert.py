#!/usr/bin/env python
"""
Bi-directional converter between Python scripts and Jupyter notebooks.

py → nb
    Converts a Python script with ``# In[*]:`` cell markers (produced by
    Jupyter's "Download as .py" / nbconvert export) back into a ``.ipynb``
    notebook.

nb → py
    Converts a Jupyter ``.ipynb`` notebook into a ``.py`` script using
    nbconvert's ``PythonExporter``.

Usage
-----
    python notebook_convert.py py2nb  input.py                    # → input.ipynb
    python notebook_convert.py py2nb  input.py  -o output.ipynb   # custom output
    python notebook_convert.py nb2py  input.ipynb                 # → input.py
    python notebook_convert.py nb2py  input.ipynb -o output.py    # custom output
    python notebook_convert.py nb2py  input.ipynb --prefix tmp_   # → tmp_input.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  py → nb  helpers
# ═══════════════════════════════════════════════════════════════════════════

# ── regex that matches cell markers like  # In[1]:  or  # In[ ]:  ──────────
CELL_MARKER = re.compile(r"^# In\[.*?]:\s*$")

# ── lines that appear at the very top of nbconvert-exported scripts ─────────
HEADER_LINES = {
    re.compile(r"^#!/usr/bin/env python\s*$"),
    re.compile(r"^# coding: .+$"),
}


def _is_header_line(line: str) -> bool:
    return any(pat.match(line) for pat in HEADER_LINES)


def _is_markdown_comment_block(lines: list[str]) -> bool:
    """Heuristic: a block made *entirely* of ``# text`` lines (with the ``# ``
    prefix) is treated as a Markdown cell.  Blank lines are allowed between
    them.  Lines that look like commented-out code (``#variable = …``) are
    *not* treated as Markdown.
    """
    non_blank = [l for l in lines if l.strip()]
    if not non_blank:
        return False
    for l in non_blank:
        stripped = l.strip()
        # Must start with "# " (note the space) to count as prose.
        # Pure "#" (empty comment) is also fine.
        if stripped == "#":
            continue
        if stripped.startswith("# "):
            # Reject lines that look like commented-out code, e.g.  # x = 1
            text_after = stripped[2:]
            # Simple heuristic: if the first non-space char after "# " is a
            # letter and the rest reads like prose, accept it.
            # We accept everything here – the user can always edit the notebook.
            continue
        return False
    return True


def _strip_comment_prefix(lines: list[str]) -> list[str]:
    """Remove the ``# `` prefix that Jupyter adds when exporting Markdown
    cells to ``.py``."""
    result = []
    for l in lines:
        stripped = l.strip()
        if stripped == "#":
            result.append("")
        elif stripped.startswith("# "):
            result.append(stripped[2:])
        else:
            result.append(l)  # blank line
    return result


def _make_code_cell(source_lines: list[str], execution_count=None) -> dict:
    # Ensure every line except the last ends with \n
    source = _ensure_newlines(source_lines)
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _make_markdown_cell(source_lines: list[str]) -> dict:
    source = _ensure_newlines(source_lines)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def _ensure_newlines(lines: list[str]) -> list[str]:
    """Jupyter expects every line except the last to end with ``\\n``."""
    if not lines:
        return lines
    out = []
    for i, l in enumerate(lines):
        if i < len(lines) - 1:
            out.append(l if l.endswith("\n") else l + "\n")
        else:
            out.append(l.rstrip("\n"))
    # Drop trailing empty string if present
    if out and out[-1] == "":
        out.pop()
    return out


def _strip_surrounding_blanks(lines: list[str]) -> list[str]:
    """Remove leading and trailing blank lines from a block."""
    while lines and lines[0].strip() == "":
        lines = lines[1:]
    while lines and lines[-1].strip() == "":
        lines = lines[:-1]
    return lines


def py_to_nb(py_path: str | Path) -> dict:
    """Read a ``.py`` file and return a notebook dict (ready to ``json.dump``)."""
    py_path = Path(py_path)
    text = py_path.read_text(encoding="utf-8")
    raw_lines = text.splitlines(keepends=False)

    # ── 1. strip header lines (shebang / encoding) ─────────────────────
    lines: list[str] = []
    for l in raw_lines:
        if not lines and _is_header_line(l):
            continue  # skip header
        lines.append(l)

    # strip leading blanks after header removal
    while lines and lines[0].strip() == "":
        lines = lines[1:]

    # ── 2. split into raw blocks on every ``# In[…]:`` marker ──────────
    blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in lines:
        if CELL_MARKER.match(line):
            if current_block:
                blocks.append(current_block)
            current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    # ── 3. convert each block into a notebook cell ──────────────────────
    cells: list[dict] = []
    for block in blocks:
        block = _strip_surrounding_blanks(block)
        if not block:
            # empty cell → empty code cell
            cells.append(_make_code_cell([]))
            continue

        if _is_markdown_comment_block(block):
            md_lines = _strip_comment_prefix(block)
            cells.append(_make_markdown_cell(md_lines))
        else:
            cells.append(_make_code_cell(block))

    # ── 4. build the notebook structure ─────────────────────────────────
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


# Keep the old name as an alias for backward compatibility
convert = py_to_nb


# ═══════════════════════════════════════════════════════════════════════════
#  nb → py
# ═══════════════════════════════════════════════════════════════════════════

def nb_to_py(nb_path: str | Path) -> str:
    """Read a ``.ipynb`` notebook and return the exported Python source string.

    Pure-Python implementation — reads the ``.ipynb`` JSON directly and
    emits ``# In[N]:`` cell markers compatible with ``py_to_nb``.
    """
    nb_path = Path(nb_path)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    parts: list[str] = [
        "#!/usr/bin/env python",
        "# coding: utf-8",
        "",
    ]

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "code")
        source_lines = cell.get("source", [])
        # source can be a list of strings or a single string
        if isinstance(source_lines, str):
            source_lines = source_lines.splitlines(keepends=True)

        exec_count = cell.get("execution_count")
        marker = f"# In[{exec_count if exec_count is not None else ' '}]:"
        parts.append(marker)
        parts.append("")

        if cell_type == "markdown":
            for line in source_lines:
                line = line.rstrip("\n")
                if line == "":
                    parts.append("#")
                else:
                    parts.append(f"# {line}")
        else:
            for line in source_lines:
                parts.append(line.rstrip("\n"))

        parts.append("")
        parts.append("")

    return "\n".join(parts) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _cmd_py2nb(args) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".ipynb")
    notebook = py_to_nb(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Converted {input_path} → {output_path}  ({len(notebook['cells'])} cells)")


def _cmd_nb2py(args) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = args.prefix + input_path.stem
        output_path = input_path.with_name(stem + ".py")

    py_source = nb_to_py(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(py_source)

    print(f"Converted {input_path} → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bi-directional converter between .py scripts and .ipynb notebooks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── py2nb ───────────────────────────────────────────────────────────
    p_py2nb = subparsers.add_parser("py2nb", help="Convert .py → .ipynb")
    p_py2nb.add_argument("input", help="Path to the input .py file")
    p_py2nb.add_argument("-o", "--output", default=None,
                         help="Output .ipynb path (default: same name, .ipynb extension)")
    p_py2nb.set_defaults(func=_cmd_py2nb)

    # ── nb2py ───────────────────────────────────────────────────────────
    p_nb2py = subparsers.add_parser("nb2py", help="Convert .ipynb → .py")
    p_nb2py.add_argument("input", help="Path to the input .ipynb file")
    p_nb2py.add_argument("-o", "--output", default=None,
                         help="Output .py path (default: same name, .py extension)")
    p_nb2py.add_argument("--prefix", default="",
                         help="Prefix for output filename (default: none; e.g. 'tmp_')")
    p_nb2py.set_defaults(func=_cmd_nb2py)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

