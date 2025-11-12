"""Helper utilities for dashboard tooltip copy.

This module loads human-editable strings from ``helptext_content.ini`` so the
team can adjust KPI and table help text without touching Python syntax. The INI
format accepts ``KEY = value`` pairs, and thanks to a custom parser setup the
``#`` character is treated as literal text rather than a comment delimiter. Each
section in the INI file mirrors the attributes exposed on :class:`HELP`.
"""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Mapping

_HELP_FILE = Path(__file__).with_name("helptext_content.ini")

_DEFAULT_KPI = {
    "KPI_STUDENTS": "Unique students with Canvas enrollments included in these metrics.",
    "KPI_AVG_GRADE": "Coursewide average Canvas score across graded assignments.",
    "KPI_MEDIAN_LETTER": "Median letter grade calculated from current Canvas scores.",
    "KPI_ECHO_ENGAGEMENT": "Average Echo360 engagement percentage across all published media.",
    "KPI_FS": "Number of students whose Canvas grade is currently below 60%.",
    "KPI_ASSIGNMENT_AVG": "Mean assignment score for the class, combining all available grades.",
}

_DEFAULT_ECHO_SUMMARY_COLUMNS = {
    "Media Title": "Name of the Echo360 media item as published to students.",
    "Video Duration": "Total runtime of the media in hours:minutes:seconds.",
    "# of Unique Viewers": "Distinct students who watched this media at least once.",
    "Average View %": "Average portion of the video watched per student viewer.",
    "% of Students Viewing": "Percent of enrolled students who viewed this media.",
    "% of Video Viewed Overall": "Share of total video minutes watched across all viewers.",
}

_DEFAULT_ECHO_MODULE_COLUMNS = {
    "Module": "Canvas module that contains these Echo360 media items.",
    "Average View %": "Mean viewing percentage across all media in the module.",
    "# of Students Viewing": "Students who watched any Echo360 media within this module.",
    "Overall View %": "Combined percentage of media watched by the viewing students.",
    "# of Students": "Total students in the course for comparison to viewers.",
}

_DEFAULT_GRADEBOOK = {
    "GRADEBOOK_SUMMARY_DEFAULT": "Assignment-level metrics aggregated from Canvas gradebook API data.",
}

_DEFAULT_GRADEBOOK_MODULE_COLUMNS = {
    "Module": "Canvas module grouping these assignments.",
    "Avg % Turned In": "Average submission rate for assignments within the module.",
    "Avg Average Excluding Zeros": "Mean assignment score ignoring missing (zero) submissions.",
    "n_assignments": "Number of assignments mapped to the module.",
}

_DEFAULT_CHARTS = {
    "CHART_ECHO": "Module-level Echo360 engagement compared against the total enrolled students.",
    "CHART_GB": "Module-level gradebook performance trends across assignments.",
}

_DEFAULT_AI = {
    "AI_ANALYSIS": "Generate an AI-authored narrative that highlights notable patterns across the dashboard.",
}


def _read_config() -> ConfigParser:
    parser = ConfigParser(
        interpolation=None,
        comment_prefixes=(),
        inline_comment_prefixes=(),
        strict=False,
    )
    parser.optionxform = str  # preserve case for column names
    if _HELP_FILE.exists():
        with _HELP_FILE.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    return parser


def _merge_section(parser: ConfigParser, section: str, defaults: Mapping[str, str]) -> dict[str, str]:
    values = dict(defaults)
    if parser.has_section(section):
        for key, value in parser.items(section):
            values[key] = value.strip()
    return values


_parser = _read_config()
_kpi = _merge_section(_parser, "KPI", _DEFAULT_KPI)
_echo_summary = _merge_section(_parser, "ECHO_SUMMARY_COLUMNS", _DEFAULT_ECHO_SUMMARY_COLUMNS)
_echo_module = _merge_section(_parser, "ECHO_MODULE_COLUMNS", _DEFAULT_ECHO_MODULE_COLUMNS)
_gradebook = _merge_section(_parser, "GRADEBOOK", _DEFAULT_GRADEBOOK)
_gradebook_module = _merge_section(_parser, "GRADEBOOK_MODULE_COLUMNS", _DEFAULT_GRADEBOOK_MODULE_COLUMNS)
_charts = _merge_section(_parser, "CHARTS", _DEFAULT_CHARTS)
_ai = _merge_section(_parser, "AI", _DEFAULT_AI)


class HELP:
    """Central place to edit dashboard help copy.

    To change any tooltip, edit ``helptext_content.ini`` next to this file. Each
    attribute below corresponds to a key in that INI document. Restart the
    Streamlit app (or clear cached resources) after saving changes to reload the
    updated text.
    """

    DEFAULT: str | None = None

    # KPI tooltips
    KPI_STUDENTS = _kpi["KPI_STUDENTS"]
    KPI_AVG_GRADE = _kpi["KPI_AVG_GRADE"]
    KPI_MEDIAN_LETTER = _kpi["KPI_MEDIAN_LETTER"]
    KPI_ECHO_ENGAGEMENT = _kpi["KPI_ECHO_ENGAGEMENT"]
    KPI_FS = _kpi["KPI_FS"]
    KPI_ASSIGNMENT_AVG = _kpi["KPI_ASSIGNMENT_AVG"]

    # Echo tables
    ECHO_SUMMARY_COLUMNS = _echo_summary
    ECHO_MODULE_COLUMNS = _echo_module

    # Gradebook tables
    GRADEBOOK_SUMMARY_DEFAULT = _gradebook["GRADEBOOK_SUMMARY_DEFAULT"]
    GRADEBOOK_MODULE_COLUMNS = _gradebook_module

    # Charts
    CHART_ECHO = _charts["CHART_ECHO"]
    CHART_GB = _charts["CHART_GB"]

    # AI tab
    AI_ANALYSIS = _ai["AI_ANALYSIS"]


__all__ = ["HELP"]
