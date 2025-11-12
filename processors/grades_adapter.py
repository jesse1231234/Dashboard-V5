# processors/grades_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz


@dataclass
class GradebookTables:
    """
    Outputs consumed by the Streamlit app.

    Notes:
      - gradebook_summary_df holds FRACTIONS 0..1 (not 0..100).
      - module_assignment_metrics_df uses FRACTIONS 0..1 for both lines.
    """
    gradebook_df: pd.DataFrame                 # cleaned, de-identified per-student rows (Final Grade/Score cols preserved)
    gradebook_summary_df: pd.DataFrame         # index: ["Average","Average Excluding Zeros","% Turned In"], columns: assignment names
    module_assignment_metrics_df: pd.DataFrame # columns: Module, Avg % Turned In, Avg Average Excluding Zeros, n_assignments


# Columns that should NOT be treated as assignment columns
IDENTITY_OR_META = {
    "student", "id", "sis user id", "sis login id", "integration id", "section",
    "final grade", "current grade", "unposted final grade",
    "final score", "current score", "unposted final score",
    "final points", "current points", "unposted current score",
}


# ---------- helpers ----------

def _lower_map(cols: Iterable[str]) -> dict[str, str]:
    return {c.lower(): c for c in cols}


def _is_assignment_col(col: str) -> bool:
    c = col.strip().lower()
    if c.startswith("unnamed"):
        return False
    return c not in IDENTITY_OR_META


def _clean_assignment_header(name: str) -> str:
    """
    Strip common Canvas trailing numeric identifiers from headers:
      - "Assignment Name (1234567)"
      - "Assignment Name - 1234567"
    """
    s = str(name).strip()
    if not s:
        return s

    # Remove trailing '(digits)'
    if s.endswith(")") and "(" in s:
        i = s.rfind("(")
        digits = s[i + 1 : -1]
        if digits.isdigit() and len(digits) >= 4:
            s = s[:i].rstrip()

    # Remove trailing '- digits'
    if "-" in s:
        left, right = s.rsplit("-", 1)
        if right.strip().isdigit() and len(right.strip()) >= 4:
            s = left.strip()

    return s


def _deidentify_students(df: pd.DataFrame) -> pd.DataFrame:
    """Replace any 'Student' column with S0001… labels and drop obvious PII ids."""
    out = df.copy()
    low = _lower_map(out.columns)
    # De-identify student name if present
    if "student" in low:
        n = len(out)
        out[low["student"]] = [f"S{i+1:04d}" for i in range(n)]
    # Drop SIS/internal ids if present
    for k in ["sis user id", "sis login id", "integration id", "id"]:
        if k in low:
            out.drop(columns=[low[k]], inplace=True)
    return out


def _assignment_columns(df_with_points_row: pd.DataFrame) -> List[str]:
    """Return assignment columns from the full gradebook (header-cleaned)."""
    result: List[str] = []
    for c in df_with_points_row.columns:
        if _is_assignment_col(c):
            result.append(c)
    return result


# ---------- main builder ----------

def build_gradebook_tables(gradebook_csv_file, canvas_order_df: pd.DataFrame) -> GradebookTables:
    """
    Read the Canvas gradebook CSV and compute:
      - Per-assignment percentage (earned / points possible)
      - Summary rows: Average, Average Excluding Zeros, % Turned In
      - Module-level averages by fuzzy matching Canvas assignments to gradebook columns

    Assumptions:
      - Row 0 is "Points Possible".
      - Rows 1..N are students (we’ll de-identify).
      - Percentages returned as fractions 0..1.
    """
    if isinstance(gradebook_csv_file, pd.DataFrame):
        gb_raw = gradebook_csv_file.copy()
    else:
        gb_raw = pd.read_csv(gradebook_csv_file)

    # Clean headers (strip trailing numeric IDs)
    gb_raw.columns = [_clean_assignment_header(c) for c in gb_raw.columns]

    if gb_raw.empty:
        return GradebookTables(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # Split points row vs. student rows
    points_row = gb_raw.iloc[0]
    students_raw = gb_raw.iloc[1:].reset_index(drop=True)

    # Remove non-student / read-only rows if they slipped in
    # (basic heuristic: drop rows where 'Student' contains "points possible" or "student, test")
    low = _lower_map(students_raw.columns)
    filt = pd.Series([True] * len(students_raw))
    if "student" in low:
        s = students_raw[low["student"]].astype(str).str.lower()
        filt &= ~s.str.contains("points possible")
        filt &= ~s.str.contains("student, test")
    students_raw = students_raw[filt].reset_index(drop=True)

    # Identify assignment columns
    assignment_cols = _assignment_columns(gb_raw)
    if not assignment_cols:
        # No assignment columns; build minimal outputs
        gradebook_df = _deidentify_students(students_raw.copy())
        gradebook_summary_df = pd.DataFrame(index=["Average", "Average Excluding Zeros", "% Turned In"])
        module_assignment_metrics_df = pd.DataFrame(columns=["Module", "Avg % Turned In", "Avg Average Excluding Zeros", "n_assignments"])
        return GradebookTables(gradebook_df, gradebook_summary_df, module_assignment_metrics_df)

    # Convert points possible + earned to numeric
    points = pd.to_numeric(points_row[assignment_cols], errors="coerce")
    earned = students_raw[assignment_cols].apply(pd.to_numeric, errors="coerce")

    # Per-assignment percentage as FRACTIONS 0..1
    perc = earned.divide(points, axis=1)

    # Summary rows
    avg = perc.mean(skipna=True)                             # includes zeros
    excl0 = perc.mask(perc == 0).mean(skipna=True)           # zeros treated as missing
    turned_in = (perc > 0).mean(skipna=True)                 # share of nonzero submissions
    gradebook_summary_df = pd.DataFrame(
        [avg, excl0, turned_in],
        index=["Average", "Average Excluding Zeros", "% Turned In"],
    )

    # De-identify student rows for any downstream use; also preserve key grade columns if present
    keep_cols = []
    for k in ["Final Grade", "Current Grade", "Unposted Final Grade", "Final Score", "Current Score", "Unposted Final Score"]:
        if k in gb_raw.columns:
            keep_cols.append(k)
    gradebook_df = students_raw[keep_cols].copy() if keep_cols else students_raw.copy()
    gradebook_df = _deidentify_students(gradebook_df)

    # ---------- module-level metrics ----------
    # Extract Canvas assignment titles per module, fuzzy-match to gradebook columns (90+)
    module_rows = []
    if {"module", "item_type", "item_title_raw"}.issubset(canvas_order_df.columns):
        canvas_assign = canvas_order_df[
            canvas_order_df["item_type"].astype(str).str.lower().str.contains("assignment", na=False)
        ]

        gb_cols = assignment_cols  # already header-cleaned

        for mod, grp in canvas_assign.groupby("module"):
            cols_for_mod: List[str] = []

            for title in grp["item_title_raw"].dropna().astype(str):
                # Clean the Canvas title similarly (helps when Canvas also appends IDs)
                cleaned_title = _clean_assignment_header(title)
                # RapidFuzz match against gradebook assignment headers
                match = process.extractOne(cleaned_title, gb_cols, scorer=fuzz.ratio)
                if match and match[1] >= 90:
                    cols_for_mod.append(match[0])

            cols_for_mod = sorted(set(cols_for_mod))
            if cols_for_mod:
                module_rows.append(
                    {
                        "Module": mod,
                        "Avg % Turned In": gradebook_summary_df.loc["% Turned In", cols_for_mod].mean(),
                        "Avg Average Excluding Zeros": gradebook_summary_df.loc["Average Excluding Zeros", cols_for_mod].mean(),
                        "n_assignments": len(cols_for_mod),
                    }
                )

    module_assignment_metrics_df = pd.DataFrame(
        module_rows, columns=["Module", "Avg % Turned In", "Avg Average Excluding Zeros", "n_assignments"]
    )

    return GradebookTables(
        gradebook_df=gradebook_df,
        gradebook_summary_df=gradebook_summary_df,
        module_assignment_metrics_df=module_assignment_metrics_df,
    )
