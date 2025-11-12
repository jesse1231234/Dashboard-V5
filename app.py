from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import streamlit as st

from services.canvas import CanvasService
from services.echo360 import Echo360Service
from processors.echo_adapter import EchoTables, build_echo_tables
from processors.grades_adapter import GradebookTables, build_gradebook_tables
from ui.charts import chart_gradebook_combo, chart_echo_combo
from ui.helptext import HELP
from ui.kpis import compute_kpis
from ai.analysis import generate_analysis
import os

st.set_page_config(page_title="Canvas/Echo Dashboard", layout="wide")
from ui.theme import apply_theme, hero
apply_theme()

hero(
    "Canvas & Echo Insights",
    "All data has been de-identified. Mouse over the '?' or the Column headers for more information about each component",
    emoji="✨",
)

# Centering toggle (wizard only)
_CSS_SLOT = st.empty()

def _set_wizard_center(on: bool):
    if on:
        _CSS_SLOT.markdown(
            """
            <style>
            /* Make the main content fill the viewport and center it vertically */
            section.main > div.block-container{
              min-height: 85vh;              /* enough height to center without clipping */
              display: flex;
              flex-direction: column;
              justify-content: center;       /* vertical centering */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        _CSS_SLOT.markdown(
            """
            <style>
            section.main > div.block-container{
              min-height: auto;
              display: block;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

NOTICE = "No identifying information will be present in this analysis. All data will be de-identified."
DEFAULT_BASE_URL = st.secrets.get("CANVAS_BASE_URL", "https://colostate.instructure.com")
DEFAULT_ECHO_BASE_URL = st.secrets.get("ECHO360_BASE_URL", "https://api.echo360.org")
TOKEN = st.secrets.get("CANVAS_TOKEN", "")
ECHO_TOKEN = st.secrets.get("ECHO360_TOKEN", "")


def render_notice(text: str, icon: str = "🔐") -> None:
    st.markdown(
        f"""
        <div class="callout">
          <span class="callout__icon">{icon}</span>
          <div class="callout__body">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_header(step: int, title: str, subtitle: str | None = None, emoji: str | None = None) -> None:
    icon = f"{emoji} " if emoji else ""
    st.markdown(
        f"""
        <div class="step-header">
          <span class="step-header__badge">{step}</span>
          <div>
            <div class="step-header__title">{icon}{title}</div>
            {f"<div class='step-header__subtitle'>{subtitle}</div>" if subtitle else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Caching helpers ----------------
@contextmanager
def get_canvas_service(base_url: str, token: str) -> Iterator[CanvasService]:
    svc = CanvasService(base_url, token)
    try:
        yield svc
    finally:
        svc.close()

@contextmanager
def get_echo_service(base_url: str, token: str) -> Iterator[Echo360Service]:
    svc = Echo360Service(base_url, token)
    try:
        yield svc
    finally:
        svc.close()


def load_canvas_data(base_url: str, token: str, course_id: str) -> tuple[pd.DataFrame, GradebookTables, Optional[int]]:
    with get_canvas_service(base_url, token) as svc:
        course_id_int = int(course_id)
        canvas_df = svc.build_order_df(course_id_int)
        gradebook_df = svc.build_gradebook_dataframe(course_id_int)
        student_count = svc.get_student_count(course_id_int)
    gradebook_tables = build_gradebook_tables(gradebook_df, canvas_df)
    return canvas_df, gradebook_tables, student_count


def load_echo_tables(
    base_url: str,
    token: str,
    section_id: str,
    canvas_df: pd.DataFrame,
    students_total: Optional[int],
 ) -> EchoTables:
    with get_echo_service(base_url, token) as svc:
        echo_df = svc.build_engagement_dataframe(section_id)
    return build_echo_tables(echo_df, canvas_df, class_total_students=students_total)

def sort_by_canvas_order(df: pd.DataFrame, module_col: str, canvas_df: pd.DataFrame) -> pd.DataFrame:
    """Sort a dataframe by Canvas module order using module_position; tolerate duplicate names."""
    if (
        df is None or df.empty or
        canvas_df is None or canvas_df.empty or
        module_col not in df.columns
    ):
        return df

    # Build ordered list of module names (keep first occurrence only)
    order = (
        canvas_df[["module", "module_position"]]
        .dropna(subset=["module"])
        .sort_values(["module_position", "module"], kind="stable")
    )
    # Deduplicate module names while preserving order
    categories = pd.unique(order["module"].astype(str))

    if len(categories) == 0:
        return df

    out = df.copy()
    out[module_col] = out[module_col].astype(str)
    out[module_col] = pd.Categorical(out[module_col], categories=categories, ordered=True)
    out = out.sort_values(module_col).reset_index(drop=True)
    # Return as string for downstream display
    out[module_col] = out[module_col].astype(str)
    return out

# --- Table display helper (place at top level, not inside another function) ---
def _percentize_for_display(
    df: pd.DataFrame,
    percent_cols: list[str],
    decimals: int = 1,
    help_text: Mapping[str, str] | str | None = HELP.DEFAULT,
):
    """
    Return (copy_of_df_with_selected_cols*100) and a Streamlit column_config for % formatting.
    """
    disp = df.copy()
    for col in percent_cols:
        if col in disp.columns:
            disp[col] = pd.to_numeric(disp[col], errors="coerce") * 100.0
    cfg: dict[str, object] = {}
    lookup: Mapping[str, str] | None
    default_help: str | None
    if isinstance(help_text, Mapping):
        lookup = help_text
        default_help = HELP.DEFAULT
    else:
        lookup = None
        default_help = help_text
    for col in disp.columns:
        help_value = lookup.get(col, default_help) if lookup else default_help
        cfg[col] = st.column_config.Column(col, help=help_value)
    for col in percent_cols:
        if col in disp.columns:
            help_value = lookup.get(col, default_help) if lookup else default_help
            cfg[col] = st.column_config.NumberColumn(
                col, format=f"%.{decimals}f%%", help=help_value
            )
    return disp, cfg

# ---------------- Wizard UI ----------------
st.session_state.setdefault("step", 1)
_state = st.session_state

with st.sidebar:
    st.markdown("### Controls")
    if st.button("Restart wizard"):
        for k in [
            "canvas",
            "echo",
            "grades",
            "results",
            "base_url",
            "course_id",
            "student_count",
            "echo_base_url",
            "echo_section_id",
        ]:
            _state.pop(k, None)
        _state.step = 1
        st.rerun()

# Center steps 1–2 (call this BEFORE rendering any step UI)
_set_wizard_center(_state.step in (1, 2))

# ---- Step 1 ----
if _state.step == 1:
    step_header(
        1,
        "Connect to your Canvas course",
        "Enter your Canvas domain and course number so we can pull module context and gradebook data automatically.",
        emoji="🧭",
    )
    render_notice(NOTICE)
    base_url = st.text_input("Canvas Base URL", value=_state.get("base_url", DEFAULT_BASE_URL))
    course_id = st.text_input(
        "Please provide the Canvas Course Number contained in the URL for the Canvas Course you are analyzing. "
        "For example, if the URL for your home page is 'https://colostate.instructure.com/courses/123456', "
        "then 123456 is your Canvas Course Number"
    )
    if not TOKEN:
        st.warning("Missing CANVAS_TOKEN in Streamlit secrets. Add it before continuing.")

    if st.button("Fetch Canvas Data") and base_url and course_id and TOKEN:
        try:
            with st.spinner("Fetching Canvas modules and gradebook data..."):
                canvas_df, gradebook_tables, student_count = load_canvas_data(base_url, TOKEN, course_id)
                _state["canvas"] = canvas_df
                _state["grades"] = gradebook_tables
                _state["base_url"] = base_url
                _state["course_id"] = course_id
                _state["student_count"] = student_count
                _state.pop("echo", None)
                _state.pop("results", None)
                _state.step = 2
                st.rerun()
        except Exception as e:
            st.error(f"Canvas error: {e}")

# ---- Step 2 ----
elif _state.step == 2:
    step_header(
        2,
        "Connect to Echo360 analytics",
        "Provide your Echo360 API details so we can retrieve engagement data automatically.",
        emoji="🎬",
    )
    render_notice(NOTICE)
    echo_base_url = st.text_input("Echo360 API Base URL", value=_state.get("echo_base_url", DEFAULT_ECHO_BASE_URL))
    echo_section_id = st.text_input("Echo360 Section ID", value=_state.get("echo_section_id", ""))

    if not ECHO_TOKEN:
        st.warning("Missing ECHO360_TOKEN in Streamlit secrets. Add it before continuing.")

    if st.button("Fetch Echo Data") and echo_base_url and echo_section_id and ECHO_TOKEN:
        try:
            with st.spinner("Fetching Echo360 analytics..."):
                echo_tables = load_echo_tables(
                    echo_base_url,
                    ECHO_TOKEN,
                    echo_section_id,
                    _state["canvas"],
                    _state.get("student_count"),
                )
                _state["echo"] = echo_tables
                _state["echo_base_url"] = echo_base_url
                _state["echo_section_id"] = echo_section_id
                _state["results"] = True
                _state.step = 3
                st.rerun()
        except Exception as e:
            st.error(f"Echo processing error: {e}")

# ---------------- Dashboard ----------------
if st.session_state.get("results"):
    st.divider()
    st.markdown("### 📊 Dashboard overview")
    echo_tables = st.session_state["echo"]
    gb_tables = st.session_state["grades"]
    canvas_df = st.session_state["canvas"]

    # Order by Canvas module order
    if hasattr(gb_tables, "module_assignment_metrics_df") and not gb_tables.module_assignment_metrics_df.empty:
        gb_tables.module_assignment_metrics_df = sort_by_canvas_order(
            gb_tables.module_assignment_metrics_df, "Module", canvas_df
        )
    if hasattr(echo_tables, "module_table") and not echo_tables.module_table.empty:
        echo_tables.module_table = sort_by_canvas_order(
            echo_tables.module_table, "Module", canvas_df
        )

    # KPIs (prefer Canvas student count when available)
    kpis = compute_kpis(
        echo_tables,
        gb_tables,
        students_from_canvas=st.session_state.get("student_count"),
    )
    
    # --- NEW: Constant "# of Students" across modules for the Echo chart ---
    students_total = (
        kpis.get("# Students")
        or st.session_state.get("student_count")
        or (len(gb_tables.gradebook_df.index) if getattr(gb_tables, "gradebook_df", None) is not None else None)
    )

    if students_total and hasattr(echo_tables, "module_table") and not echo_tables.module_table.empty:
        echo_tables.module_table = echo_tables.module_table.copy()
        echo_tables.module_table["# of Students"] = int(students_total)


    # KPI header
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("# Students", f"{kpis.get('# Students', 0):,}", help=HELP.KPI_STUDENTS)
    avg_grade = kpis.get("Average Grade")
    c2.metric("Median Letter Grade", kpis.get("Median Letter Grade", "—"), help=HELP.KPI_MEDIAN_LETTER)
    avg_echo = kpis.get("Average Echo360 engagement")
    c3.metric("Avg Echo Engagement", f"{avg_echo:.1f}%" if avg_echo is not None else "—", help=HELP.KPI_ECHO_ENGAGEMENT)
    c4.metric("# of Fs", f"{kpis.get('# of Fs', 0):,}", help=HELP.KPI_FS)
    avg_assign = kpis.get("Avg Assignment Grade (class)")
    c5.metric(
        "Avg Assignment Grade",
        f"{avg_assign*100:.1f}%" if avg_assign is not None else "—",
        help=HELP.KPI_ASSIGNMENT_AVG,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Tables", "Charts", "Exports", "AI Analysis"])

    with tab1:
        st.subheader("Echo Summary (per media)")
        es_disp, es_cfg = _percentize_for_display(
            echo_tables.echo_summary,
            ["Average View %", "% of Students Viewing", "% of Video Viewed Overall"],
            help_text=HELP.ECHO_SUMMARY_COLUMNS,
        )
        st.dataframe(es_disp, width="stretch", column_config=es_cfg)


        st.subheader("Echo Module Table")
        em_disp, em_cfg = _percentize_for_display(
            echo_tables.module_table,
            ["Average View %", "Overall View %"],
            help_text=HELP.ECHO_MODULE_COLUMNS,
        )
        st.dataframe(em_disp, width="stretch", column_config=em_cfg)


        st.subheader("Gradebook Summary Rows")
        gb_sum_disp = gb_tables.gradebook_summary_df.copy()
        # all columns are fractions → scale to %
        gb_sum_disp = gb_sum_disp.apply(pd.to_numeric, errors="coerce") * 100.0
        gb_sum_cfg = {
            col: st.column_config.NumberColumn(col, format="%.1f%%", help=HELP.GRADEBOOK_SUMMARY_DEFAULT)
            for col in gb_sum_disp.columns
        }
        st.dataframe(gb_sum_disp, width="stretch", column_config=gb_sum_cfg)


        st.subheader("Gradebook Module Metrics")
        gm_disp, gm_cfg = _percentize_for_display(
            gb_tables.module_assignment_metrics_df,
            ["Avg % Turned In", "Avg Average Excluding Zeros"],
            help_text=HELP.GRADEBOOK_MODULE_COLUMNS,
        )
        st.dataframe(gm_disp, width="stretch", column_config=gm_cfg)

    with tab2:
        if not gb_tables.module_assignment_metrics_df.empty:
            st.plotly_chart(
                chart_gradebook_combo(gb_tables.module_assignment_metrics_df, title="Canvas Data"),
                width="stretch",
            )
            st.caption(f"ℹ️ {HELP.CHART_GB}")
        else:
            st.info("No module-level gradebook metrics to plot.")

        if not echo_tables.module_table.empty:
            st.plotly_chart(
                chart_echo_combo(echo_tables.module_table, students_total=students_total, title="Echo Data"),
                width="stretch"
            )
            st.caption(f"ℹ️ {HELP.CHART_ECHO}")

        else:
            st.info("No module-level Echo metrics to plot.")

    with tab3:
        def to_csv_bytes(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Echo Summary CSV",
            to_csv_bytes(echo_tables.echo_summary),
            file_name="echo_summary.csv",
        )
        st.download_button(
            "Download Echo Module Table CSV",
            to_csv_bytes(echo_tables.module_table),
            file_name="echo_module_table.csv",
        )
        st.download_button(
            "Download Gradebook Summary CSV",
            gb_tables.gradebook_summary_df.to_csv().encode("utf-8"),
            file_name="gradebook_summary.csv",
        )
        st.download_button(
            "Download Gradebook Module Metrics CSV",
            to_csv_bytes(gb_tables.module_assignment_metrics_df),
            file_name="gradebook_module_metrics.csv",
        )

    with tab4:
        st.subheader("AI Analysis")
        st.caption(f"ℹ️ {HELP.AI_ANALYSIS}")
        st.caption("No identifying information will be present in this analysis. All data will be de-identified.")

        # Model + settings
        colA, colB = st.columns([2,1])
        with colA:
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
        with colB:
            temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)

        # Confirm key present (use st.secrets or env)
        openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not openai_key:
            st.warning("Add OPENAI_API_KEY to Streamlit secrets to enable AI analysis.")
        else:
            # Button to generate
            if st.button("Generate analysis"):
                with st.spinner("Analyzing your dashboard data..."):
                    try:
                        text = generate_analysis(
                            kpis=kpis,
                            echo_module_df=echo_tables.module_table if echo_tables else None,
                            gradebook_module_df=gb_tables.module_assignment_metrics_df if gb_tables else None,
                            gradebook_summary_df=gb_tables.gradebook_summary_df if gb_tables else None,
                            model=model,
                            temperature=temperature,
                        )
                        st.markdown(text)
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")











