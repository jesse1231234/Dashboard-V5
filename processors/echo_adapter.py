# processors/echo_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, List, Tuple
import re

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz


@dataclass
class EchoTables:
    echo_summary: pd.DataFrame            # per-media summary
    module_table: pd.DataFrame            # per-module aggregation (ordered by Canvas)
    student_table: pd.DataFrame           # de-identified per-student engagement


# Column name candidates (case-insensitive, fuzzy 'contains' fallback)
CANDIDATES = {
    "media":     ["media name", "media title", "video title", "title", "name"],
    "duration":  ["duration", "video duration", "media duration", "length"],
    "viewtime":  ["total view time", "total viewtime", "total watch time", "view time"],
    "avgview":   ["average view time", "avg view time", "avg watch time", "average watch time"],
    "user":      ["user email", "user name", "email", "user", "viewer", "username"],
}

# Cleaning patterns
_DURATION_TAIL_RE = re.compile(r"\s*\((?:\d{1,2}:)?\d{1,2}:\d{2}\)\s*$", re.I)
_NUM_ID_TAIL_RE   = re.compile(r"\s*-\s*\d{4,}\s*$")
_READ_ONLY_RE     = re.compile(r"\s*\(read only\)\s*$", re.I)

# Fuzzy matching knobs
FUZZY_SCORER   = fuzz.token_set_ratio
THRESHOLD      = 80
FALLBACK_MIN   = 70
TOP_K          = 6


# ---------- helpers ----------

def _find_col(df: pd.DataFrame, want: Iterable[str], required: bool = True) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for w in want:
        if w in low:
            return low[w]
    for k, v in low.items():
        if any(w in k for w in want):
            return v
    if required:
        raise KeyError(f"Missing required column; need one of: {list(want)}\nAvailable: {list(df.columns)}")
    return None


def _to_seconds(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except Exception:
        pass
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except Exception:
        return np.nan
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    return np.nan


def _strip_noise_tail(title: str) -> str:
    if not title:
        return ""
    s = str(title)
    s = _READ_ONLY_RE.sub("", s)
    s = _DURATION_TAIL_RE.sub("", s)
    s = _NUM_ID_TAIL_RE.sub("", s)
    return s.strip()


def _norm_text(text: str) -> str:
    s = _strip_noise_tail(text)
    s = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def _norm_series(s: pd.Series) -> pd.Series:
    return s.fillna("").map(_norm_text)


def _greedy_match(
    ekeys: List[str],
    ckeys: List[str],
    threshold: int,
    fallback_min: int,
    top_k: int,
) -> List[Tuple[int, int, int]]:
    candidates: List[Tuple[int, int, int]] = []
    for i, ek in enumerate(ekeys):
        topn = process.extract(ek, ckeys, scorer=FUZZY_SCORER, limit=top_k)
        for _, sc, j in topn:
            if sc >= threshold:
                candidates.append((i, j, int(sc)))

    if not candidates:
        for i, ek in enumerate(ekeys):
            best = process.extractOne(ek, ckeys, scorer=FUZZY_SCORER)
            if best and best[1] >= fallback_min:
                candidates.append((i, int(best[2]), int(best[1])))

    candidates.sort(key=lambda t: t[2], reverse=True)
    used_e: set[int] = set()
    used_c: set[int] = set()
    chosen: List[Tuple[int, int, int]] = []
    for i, j, sc in candidates:
        if i in used_e or j in used_c:
            continue
        chosen.append((i, j, sc))
        used_e.add(i)
        used_c.add(j)
    return chosen


# ---------- main builder ----------

def build_echo_tables(
    echo_csv_file,
    canvas_order_df: pd.DataFrame,
    class_total_students: Optional[int] = None,   # <--- NEW
) -> EchoTables:
    """
    Build Echo tables.

    Fractions are in [0..1]; charts display them as 0..100%.
    """
    if isinstance(echo_csv_file, pd.DataFrame):
        df = echo_csv_file.copy()
    else:
        df = pd.read_csv(echo_csv_file)

    media_col = _find_col(df, CANDIDATES["media"], required=True)
    dur_col   = _find_col(df, CANDIDATES["duration"], required=True)
    view_col  = _find_col(df, CANDIDATES["viewtime"], required=True)
    avgv_col  = _find_col(df, CANDIDATES["avgview"], required=False)
    uid_col   = _find_col(df, CANDIDATES["user"],   required=False)

    # Normalize time columns to seconds
    df[dur_col]  = df[dur_col].map(_to_seconds)
    df[view_col] = df[view_col].map(_to_seconds)
    if avgv_col:
        df[avgv_col] = df[avgv_col].map(_to_seconds)

    # Row-level true view fraction (0..1)
    df["__true_view_frac"] = np.where(df[dur_col] > 0, df[view_col] / df[dur_col], np.nan)

    # ---------- per-media summary ----------
    g = df.groupby(df[media_col].astype(str))
    # Unique viewers (prefer user id; else count non-null rows)
    if uid_col:
        uniq_viewers = g[uid_col].nunique(dropna=True)
    else:
        uniq_viewers = g[view_col].apply(lambda s: s.notna().sum())

    # Representative title & duration per media
    title_per_media = g[media_col].agg(lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else "")
    dur_per_media = g[dur_col].first()

    # Sum of all view seconds per media (for 'Overall View %')
    sum_view_seconds = g[view_col].sum(min_count=1)

    echo_summary = pd.DataFrame({
        "Media Title": title_per_media.index,
        "Video Duration": dur_per_media.reindex(title_per_media.index).values,
        "# of Unique Viewers": uniq_viewers.reindex(title_per_media.index).fillna(0).astype(int).values,
        "Average View %": g["__true_view_frac"].mean().reindex(title_per_media.index).values,   # 0..1
    })

    # % of Students Viewing (media-level)
    if class_total_students and class_total_students > 0:
        echo_summary["% of Students Viewing"] = (
            echo_summary["# of Unique Viewers"].astype(float) / float(class_total_students)
        )
    else:
        echo_summary["% of Students Viewing"] = np.nan

    # % of Video Viewed Overall (media-level)
    if class_total_students and class_total_students > 0:
        denom = echo_summary["Video Duration"].astype(float) * float(class_total_students)
        with np.errstate(divide="ignore", invalid="ignore"):
            echo_summary["% of Video Viewed Overall"] = (
                sum_view_seconds.reindex(title_per_media.index).to_numpy() / denom.to_numpy()
            )
    else:
        echo_summary["% of Video Viewed Overall"] = np.nan

    # ---------- Canvas join ----------
    module_col = "module"
    if canvas_order_df is None or canvas_order_df.empty or module_col not in canvas_order_df.columns:
        module_table = pd.DataFrame(columns=[
            "Module", "Average View %", "# of Students Viewing", "Overall View %", "# of Students"
        ])
    else:
        # Prefer duration-stripped Canvas title if available
        canvas_title_col = None
        for col in ["video_title_raw", "item_title_raw", "item_title_normalized"]:
            if col in canvas_order_df.columns:
                canvas_title_col = col
                break

        order = (
            canvas_order_df[[module_col, "module_position", canvas_title_col]]
            .dropna(subset=[module_col, canvas_title_col])
            .rename(columns={canvas_title_col: "Canvas Title"})
            .copy()
        )

        order["_ckey"] = _norm_series(order["Canvas Title"])
        es = echo_summary.copy()
        es["_ekey"] = _norm_series(es["Media Title"])

        # 1) Exact key equality
        m1 = order.merge(
            es[["_ekey", "Media Title", "Video Duration", "# of Unique Viewers", "Average View %", "% of Video Viewed Overall"]],
            left_on="_ckey", right_on="_ekey", how="left"
        )

        # 2) Greedy one-to-one fuzzy matching for still-unmatched rows
        unmatched_idx = m1.index[m1["Average View %"].isna()].tolist()
        if unmatched_idx:
            ckeys = m1.loc[unmatched_idx, "_ckey"].fillna("").astype(str).tolist()
            ekeys = es["_ekey"].fillna("").astype(str).tolist()
            pairs = _greedy_match(ekeys, ckeys, THRESHOLD, FALLBACK_MIN, TOP_K)
            if pairs:
                rows: List[dict] = []
                for i, j, sc in pairs:
                    m1_row_index = unmatched_idx[j]
                    ek = ekeys[i]
                    erow = es.loc[es["_ekey"] == ek].iloc[0]
                    rows.append({
                        "idx": m1_row_index,
                        "Media Title": erow["Media Title"],
                        "Video Duration": erow["Video Duration"],
                        "# of Unique Viewers": erow["# of Unique Viewers"],
                        "Average View %": erow["Average View %"],
                        "% of Video Viewed Overall": erow["% of Video Viewed Overall"],
                    })
                if rows:
                    m2 = pd.DataFrame(rows).set_index("idx")
                    fill_cols = ["Media Title", "Video Duration", "# of Unique Viewers", "Average View %", "% of Video Viewed Overall"]
                    m1.loc[m2.index, fill_cols] = m2[fill_cols].values

        # Aggregate by module:
        # - Average View %  => mean of the media in module
        # - # of Students Viewing => mean of media-level "# of Unique Viewers"
        # - Overall View %  => mean of media-level "% of Video Viewed Overall"
        have = m1.dropna(subset=["Average View %"])
        if have.empty:
            module_table = pd.DataFrame(columns=[
                "Module", "Average View %", "# of Students Viewing", "Overall View %", "# of Students"
            ])
        else:
            module_table = (
                have.groupby([module_col, "module_position"], as_index=False)
                    .agg({
                        "Average View %": "mean",
                        "# of Unique Viewers": "mean",
                        "% of Video Viewed Overall": "mean",
                    })
                    .rename(columns={
                        "# of Unique Viewers": "# of Students Viewing",
                        "% of Video Viewed Overall": "Overall View %",
                        module_col: "Module"
                    })
                    .sort_values(["module_position"])
                    .drop(columns=["module_position"])
            )
            # Fill constant # of Students if provided (helps UI)
            if class_total_students:
                module_table["# of Students"] = int(class_total_students)
            else:
                module_table["# of Students"] = np.nan

    # ---------- de-identified student summary ----------
    if uid_col:
        student = (
            df.assign(_frac=df["__true_view_frac"])
              .groupby(df[uid_col].fillna("unknown"))
              .agg(**{"Average View % When Watched": ("_frac", "mean")})
              .reset_index(drop=False)
              .rename(columns={uid_col: "Student"})
        )
        total_seconds = df[dur_col].dropna().astype(float).sum()
        per_user_seconds = df.groupby(uid_col)[view_col].sum(min_count=1)
        student["View % of Total Video"] = (
            per_user_seconds.reindex(student["Student"]).to_numpy() / total_seconds
            if total_seconds else np.nan
        )
        student["Student"] = [f"S{ix+1:04d}" for ix in range(len(student))]
        student_table = student[["Student", "Average View % When Watched", "View % of Total Video"]]
        student_table["Final Grade"] = np.nan
    else:
        student_table = pd.DataFrame(columns=[
            "Student", "Final Grade", "Average View % When Watched", "View % of Total Video"
        ])

    if "__true_view_frac" in df.columns:
        del df["__true_view_frac"]

    return EchoTables(
        echo_summary=echo_summary,
        module_table=module_table,
        student_table=student_table,
    )
