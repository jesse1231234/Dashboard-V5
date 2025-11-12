from __future__ import annotations

from typing import Dict, List, Optional, Callable

import httpx
import pandas as pd


class Echo360Service:
    """Lightweight client for Echo360 analytics endpoints."""

    def __init__(self, base_url: str, token: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    # ---------------- Internal helpers ----------------

    @staticmethod
    def _next_link(link_header: Optional[str]) -> Optional[str]:
        if not link_header:
            return None
        for part in link_header.split(","):
            seg = part.strip()
            if "rel=\"next\"" in seg.lower():
                url_part = seg.split(";")[0].strip()
                return url_part.strip("<>")
        return None

    @staticmethod
    def _extract_items(payload) -> List[Dict]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("results", "data", "items", "analytics", "content", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        return []

    def _get_all(self, url: str, params: Dict | None = None) -> List[Dict]:
        out: List[Dict] = []
        next_url = url
        next_params = params or {}
        while next_url:
            resp = self.client.get(next_url, params=next_params)
            resp.raise_for_status()
            items = self._extract_items(resp.json())
            out.extend(items)
            next_url = self._next_link(resp.headers.get("Link"))
            next_params = None
        return out

    def _fetch_with_fallback(self, builder: Callable[[str], str]) -> List[Dict]:
        """Attempt a sections endpoint first; on 404 retry a courses endpoint."""
        url = builder("sections")
        try:
            return self._get_all(url, params={"per_page": 100})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            url = builder("courses")
            return self._get_all(url, params={"per_page": 100})

    # ---------------- Public API ----------------

    def get_media_summaries(self, section_id: str) -> List[Dict]:
        return self._fetch_with_fallback(
            lambda resource: f"{self.base_url}/api/v1/{resource}/{section_id}/analytics/media"
        )

    def get_viewer_engagement(self, section_id: str) -> List[Dict]:
        return self._fetch_with_fallback(
            lambda resource: f"{self.base_url}/api/v1/{resource}/{section_id}/analytics/viewers"
        )

    def build_engagement_dataframe(self, section_id: str) -> pd.DataFrame:
        """Return a DataFrame mirroring the legacy Echo CSV columns."""
        try:
            viewer_records = self.get_viewer_engagement(section_id)
        except httpx.HTTPStatusError:
            viewer_records = []

        rows: List[Dict[str, object]] = []
        for rec in viewer_records:
            if not isinstance(rec, dict):
                continue
            media = rec.get("media") if isinstance(rec, dict) else None
            viewer = rec.get("viewer") if isinstance(rec, dict) else None
            rows.append(
                {
                    "Media Title": (
                        (media or {}).get("title")
                        or (media or {}).get("name")
                        or rec.get("mediaTitle")
                        or rec.get("media_name")
                        or rec.get("title")
                    ),
                    "Video Duration": (
                        (media or {}).get("durationSeconds")
                        or rec.get("durationSeconds")
                        or rec.get("mediaDuration")
                        or rec.get("duration")
                    ),
                    "Total View Time": (
                        rec.get("viewSeconds")
                        or rec.get("viewTimeSeconds")
                        or rec.get("totalViewSeconds")
                        or rec.get("viewTime")
                    ),
                    "Average View Time": (
                        rec.get("averageViewSeconds")
                        or rec.get("avgViewSeconds")
                        or rec.get("averageViewTimeSeconds")
                        or rec.get("averageViewTime")
                        or rec.get("viewSeconds")
                    ),
                    "User Email": (
                        (viewer or {}).get("email")
                        or (viewer or {}).get("username")
                        or rec.get("viewerEmail")
                        or rec.get("viewer")
                    ),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            return df

        try:
            media_records = self.get_media_summaries(section_id)
        except httpx.HTTPStatusError:
            media_records = []

        agg_rows: List[Dict[str, object]] = []
        for rec in media_records:
            if not isinstance(rec, dict):
                continue
            media = rec.get("media") if isinstance(rec, dict) else None
            analytics = rec.get("analytics") if isinstance(rec, dict) else None
            agg_rows.append(
                {
                    "Media Title": (
                        (media or {}).get("title")
                        or (media or {}).get("name")
                        or rec.get("title")
                    ),
                    "Video Duration": (
                        (media or {}).get("durationSeconds")
                        or rec.get("durationSeconds")
                        or (analytics or {}).get("durationSeconds")
                        or rec.get("mediaDuration")
                    ),
                    "Total View Time": (
                        rec.get("totalViewSeconds")
                        or (analytics or {}).get("totalViewSeconds")
                    ),
                    "Average View Time": (
                        rec.get("averageViewSeconds")
                        or (analytics or {}).get("averageViewSeconds")
                    ),
                    "User Email": None,
                }
            )

        return pd.DataFrame(agg_rows, columns=[
            "Media Title",
            "Video Duration",
            "Total View Time",
            "Average View Time",
            "User Email",
        ])

    # ---------------- Cleanup ----------------

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
