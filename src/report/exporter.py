"""
report/exporter.py  (Module 15: Report Export)
==============================================
Exports analytics to CSV and PDF. CSV uses pandas (always available). PDF uses
fpdf2 if installed; if not, the function returns a clearly-labelled plain-text
report as bytes so export never silently fails.
"""

from __future__ import annotations

import io
from typing import Dict

import pandas as pd

try:
    from fpdf import FPDF  # fpdf2
    from fpdf.enums import XPos, YPos
    _FPDF_AVAILABLE = True
except Exception:                       # pragma: no cover
    _FPDF_AVAILABLE = False


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _clean(text: str) -> str:
    """fpdf core fonts are latin-1; replace unsupported chars defensively."""
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_report(title: str, sections: Dict[str, str]) -> bytes:
    """
    sections: ordered dict of {heading: body_text}. Returns PDF bytes (fpdf2)
    or a plain-text fallback encoded as bytes.
    """
    if not _FPDF_AVAILABLE:
        lines = [title, "=" * len(title), ""]
        for h, body in sections.items():
            lines += [h, "-" * len(h), body, ""]
        return ("\n".join(lines)).encode("utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _clean(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    for heading, body in sections.items():
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, _clean(heading), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _clean(body) or "Cannot determine from provided commentary.",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")


def build_scorecard_csv(batting: pd.DataFrame, bowling: pd.DataFrame) -> bytes:
    """Combine batting + bowling into one CSV with section markers."""
    buf = io.StringIO()
    buf.write("BATTING\n")
    if not batting.empty:
        batting.to_csv(buf, index=False)
    buf.write("\nBOWLING\n")
    if not bowling.empty:
        bowling.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
