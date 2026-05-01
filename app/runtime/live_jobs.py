from __future__ import annotations

import html
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

SERPDOG_ENDPOINT = "https://api.serpdog.io/linkedin_jobs"
HIMALAYAS_ENDPOINT = "https://himalayas.app/jobs/api/search"
REMOTIVE_ENDPOINT = "https://remotive.com/api/remote-jobs"
ARBEITNOW_ENDPOINT = "https://www.arbeitnow.com/api/job-board-api"
REMOTEOK_ENDPOINT = "https://remoteok.com/api"
DEFAULT_LINKEDIN_GEO_ID = "103644278"
DEFAULT_SORT_BY = "week"
MAX_QUERY_CHARS = 120
MAX_LIVE_CANDIDATES = 25
DEFAULT_LIVE_TOP_K = 5
KEYLESS_PROVIDER_LIMIT = 80
_LIVE_QUERY_STOPWORDS = {
    "associate",
    "director",
    "engineer",
    "junior",
    "lead",
    "manager",
    "mid",
    "principal",
    "senior",
    "staff",
}


def serpdog_api_key() -> str:
    return os.environ.get("SERPDOG_API_KEY", "").strip()


def linkedin_geo_id() -> str:
    return os.environ.get("LINKEDIN_GEO_ID", DEFAULT_LINKEDIN_GEO_ID).strip()


def build_live_job_query(
    profile: dict[str, Any] | None,
    cluster: dict[str, Any] | None,
    matches: pd.DataFrame | None,
) -> str:
    """Build a compact live-jobs query from model outputs, not raw resume text."""
    parts: list[str] = []
    if profile:
        _append_phrase(parts, profile.get("track"))
        _append_phrase(parts, profile.get("seniority"))
        for skill in (profile.get("skills_present") or [])[:3]:
            _append_phrase(parts, skill)

    if matches is not None and not matches.empty and "title" in matches.columns:
        _append_phrase(parts, matches.iloc[0].get("title"))

    if cluster:
        _append_phrase(parts, cluster.get("label"))
        for term in (cluster.get("top_terms") or [])[:3]:
            _append_phrase(parts, term)

    return _cap_query(" ".join(parts), max_chars=MAX_QUERY_CHARS)


def exp_level_for_seniority(seniority: str | None) -> str | None:
    text = str(seniority or "").lower()
    if any(token in text for token in ("intern", "entry", "junior")):
        return "entry_level"
    if "associate" in text:
        return "associate"
    if any(token in text for token in ("lead", "executive", "director", "vp")):
        return "director"
    if any(token in text for token in ("mid", "senior", "staff", "principal")):
        return "mid_senior_level"
    return None


def fetch_serpdog_linkedin_jobs(
    query: str,
    *,
    api_key: str,
    geo_id: str = DEFAULT_LINKEDIN_GEO_ID,
    exp_level: str | None = None,
    sort_by: str = DEFAULT_SORT_BY,
    timeout: float = 8.0,
    endpoint: str = SERPDOG_ENDPOINT,
    opener: Callable[[str, float], bytes] | None = None,
) -> pd.DataFrame:
    """Fetch one page of live LinkedIn jobs through Serpdog.

    Failures return an empty DataFrame so the UI can hide the live section.
    """
    query = query.strip()
    api_key = api_key.strip()
    geo_id = str(geo_id or DEFAULT_LINKEDIN_GEO_ID).strip()
    if not query or not api_key:
        return empty_live_jobs_frame("Missing query or SERPDOG_API_KEY.")

    params = {
        "api_key": api_key,
        "q": query,
        "geoId": geo_id or DEFAULT_LINKEDIN_GEO_ID,
        "sort_by": sort_by,
        "page": "0",
    }
    if exp_level:
        params["exp_level"] = exp_level

    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    try:
        raw = (
            opener(url, timeout) if opener is not None else _default_open(url, timeout)
        )
        payload = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return empty_live_jobs_frame(f"Serpdog returned HTTP {exc.code}.")
    except (
        TimeoutError,
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        return empty_live_jobs_frame(f"Live lookup failed: {type(exc).__name__}.")

    return parse_serpdog_jobs(payload)


def fetch_live_jobs(
    query: str,
    *,
    serpdog_key: str = "",
    geo_id: str = DEFAULT_LINKEDIN_GEO_ID,
    exp_level: str | None = None,
    sort_by: str = DEFAULT_SORT_BY,
    timeout: float = 8.0,
    opener: Callable[[str, float], bytes] | None = None,
) -> pd.DataFrame:
    """Fetch live jobs from no-key sources plus optional Serpdog LinkedIn."""
    query = query.strip()
    if not query:
        return empty_live_jobs_frame("No live-search query could be built.")

    frames = [
        fetch_himalayas_jobs(query, timeout=timeout, opener=opener),
        fetch_remotive_jobs(query, timeout=timeout, opener=opener),
        fetch_arbeitnow_jobs(query, timeout=timeout, opener=opener),
        fetch_remoteok_jobs(query, timeout=timeout, opener=opener),
    ]
    if serpdog_key.strip():
        frames.append(
            fetch_serpdog_linkedin_jobs(
                query,
                api_key=serpdog_key,
                geo_id=geo_id,
                exp_level=exp_level,
                sort_by=sort_by,
                timeout=timeout,
                opener=opener,
            )
        )

    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        reasons = [
            str(frame.attrs.get("reason", "")).strip()
            for frame in frames
            if frame is not None and frame.attrs.get("reason")
        ]
        reason = (
            "; ".join(reasons[:3])
            or "No keyless live job providers returned usable rows."
        )
        return empty_live_jobs_frame(reason)

    combined = pd.concat(usable, ignore_index=True)
    combined = combined.drop_duplicates(subset=["job_link"], keep="first")
    if "job_id" in combined.columns:
        has_job_id = combined["job_id"].astype(str).str.strip().ne("")
        with_ids = combined.loc[has_job_id].drop_duplicates(
            subset=["source", "job_id"], keep="first"
        )
        combined = pd.concat([with_ids, combined.loc[~has_job_id]], ignore_index=True)
    return _ensure_live_columns(
        combined.head(KEYLESS_PROVIDER_LIMIT).reset_index(drop=True)
    )


def fetch_himalayas_jobs(
    query: str,
    *,
    timeout: float = 8.0,
    opener: Callable[[str, float], bytes] | None = None,
) -> pd.DataFrame:
    params = {"q": query, "sort": "recent", "page": "1"}
    return _fetch_provider_json(
        f"{HIMALAYAS_ENDPOINT}?{urllib.parse.urlencode(params)}",
        parse_himalayas_jobs,
        "Himalayas",
        timeout,
        opener,
    )


def fetch_remotive_jobs(
    query: str,
    *,
    timeout: float = 8.0,
    opener: Callable[[str, float], bytes] | None = None,
) -> pd.DataFrame:
    params = {"search": query, "limit": "25"}
    return _fetch_provider_json(
        f"{REMOTIVE_ENDPOINT}?{urllib.parse.urlencode(params)}",
        parse_remotive_jobs,
        "Remotive",
        timeout,
        opener,
    )


def fetch_arbeitnow_jobs(
    query: str,
    *,
    timeout: float = 8.0,
    opener: Callable[[str, float], bytes] | None = None,
) -> pd.DataFrame:
    frame = _fetch_provider_json(
        ARBEITNOW_ENDPOINT,
        parse_arbeitnow_jobs,
        "Arbeitnow",
        timeout,
        opener,
    )
    return _local_filter_provider_frame(frame, query)


def fetch_remoteok_jobs(
    query: str,
    *,
    timeout: float = 8.0,
    opener: Callable[[str, float], bytes] | None = None,
) -> pd.DataFrame:
    frame = _fetch_provider_json(
        REMOTEOK_ENDPOINT,
        parse_remoteok_jobs,
        "RemoteOK",
        timeout,
        opener,
    )
    return _local_filter_provider_frame(frame, query)


def parse_serpdog_jobs(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    job_results = payload.get("job_results", [])
    if not isinstance(job_results, list):
        return empty_live_jobs_frame("Serpdog response did not include job_results.")

    for idx, item in enumerate(job_results, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("job_position") or "").strip()
        link = str(item.get("job_link") or "").strip()
        if not title or not _is_linkedin_job_url(link):
            continue
        rows.append(
            {
                "provider_rank": idx,
                "title": title,
                "company_name": str(item.get("company_name") or "").strip(),
                "location": str(item.get("job_location") or "").strip(),
                "posting_date": str(item.get("job_posting_date") or "").strip(),
                "job_link": link,
                "job_id": str(item.get("job_id") or "").strip(),
                "company_profile": str(item.get("company_profile") or "").strip(),
                "snippet": "",
                "source": "Serpdog LinkedIn Jobs",
            }
        )

    if not rows:
        return empty_live_jobs_frame("Serpdog returned no usable LinkedIn job rows.")
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["job_link"], keep="first")
    if "job_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["job_id"], keep="first")
    return _ensure_live_columns(frame.reset_index(drop=True))


def rank_live_jobs(
    live_jobs: pd.DataFrame,
    encoder: Any,
    resume_embedding: np.ndarray,
    query_text: str,
    *,
    top_k: int = DEFAULT_LIVE_TOP_K,
    max_candidates: int = MAX_LIVE_CANDIDATES,
    today: date | None = None,
) -> pd.DataFrame:
    if live_jobs is None or live_jobs.empty:
        return empty_live_jobs_frame("No live candidates to rerank.")

    candidates = _ensure_live_columns(live_jobs).head(max_candidates).copy()
    texts = [_live_embedding_text(row) for _, row in candidates.iterrows()]
    if not texts:
        return empty_live_jobs_frame("No live candidate text to rerank.")

    embeddings = np.asarray(encoder.encode(texts), dtype=np.float32)
    resume_vec = np.asarray(resume_embedding, dtype=np.float32).reshape(1, -1)
    similarities = _cosine_similarity(resume_vec, embeddings)
    query_tokens = _token_set(query_text)

    overlap = [_token_overlap(query_tokens, _token_set(text)) for text in texts]
    freshness = [
        freshness_score(value, today=today)
        for value in candidates["posting_date"].astype(str).tolist()
    ]

    normalized_similarity = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
    final = (
        0.80 * normalized_similarity
        + 0.10 * np.asarray(overlap, dtype=np.float32)
        + 0.10 * np.asarray(freshness, dtype=np.float32)
    )

    candidates["embedding_score"] = normalized_similarity
    candidates["token_overlap"] = overlap
    candidates["freshness_score"] = freshness
    candidates["live_match_score"] = np.round(final * 100.0, 2)
    return (
        candidates.sort_values(
            ["live_match_score", "provider_rank"], ascending=[False, True]
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def freshness_score(value: str, *, today: date | None = None) -> float:
    parsed = _parse_date(value)
    if parsed is None:
        return 0.3
    current = today or date.today()
    age = max(0, (current - parsed).days)
    if age <= 7:
        return 1.0
    if age <= 14:
        return 0.75
    if age <= 30:
        return 0.5
    if age <= 60:
        return 0.25
    return 0.0


def empty_live_jobs_frame(reason: str = "") -> pd.DataFrame:
    frame = pd.DataFrame(
        columns=[
            "provider_rank",
            "title",
            "company_name",
            "location",
            "posting_date",
            "job_link",
            "job_id",
            "company_profile",
            "snippet",
            "source",
            "embedding_score",
            "token_overlap",
            "freshness_score",
            "live_match_score",
        ]
    )
    if reason:
        frame.attrs["reason"] = reason
    return frame


def _default_open(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "resumatch/1.0"})
    with urllib.request.urlopen(
        request,
        timeout=timeout,
        context=_ssl_context(),
    ) as response:
        status = getattr(response, "status", 200)
        if int(status) >= 400:
            raise urllib.error.HTTPError(url, status, "Live provider error", {}, None)
        return response.read()


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except (ImportError, OSError):
        return ssl.create_default_context()


def _fetch_provider_json(
    url: str,
    parser: Callable[[dict[str, Any] | list[Any]], pd.DataFrame],
    provider: str,
    timeout: float,
    opener: Callable[[str, float], bytes] | None,
) -> pd.DataFrame:
    try:
        raw = (
            opener(url, timeout) if opener is not None else _default_open(url, timeout)
        )
        payload = json.loads(raw.decode("utf-8"))
    except (
        TimeoutError,
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        return empty_live_jobs_frame(f"{provider} lookup failed: {type(exc).__name__}.")
    frame = parser(payload)
    if frame.empty and not frame.attrs.get("reason"):
        frame.attrs["reason"] = f"{provider} returned no usable rows."
    return frame


def parse_himalayas_jobs(payload: dict[str, Any] | list[Any]) -> pd.DataFrame:
    items = _payload_items(payload, "jobs")
    rows = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        link = str(item.get("applicationLink") or "").strip()
        if not title or not link:
            continue
        rows.append(
            _live_row(
                provider_rank=idx,
                title=title,
                company=item.get("companyName"),
                location=", ".join(item.get("locationRestrictions") or []) or "Remote",
                posting_date=item.get("pubDate"),
                link=link,
                job_id=item.get("guid"),
                source="Himalayas",
                snippet=item.get("excerpt") or _strip_html(item.get("description")),
            )
        )
    return _frame_from_rows(rows)


def parse_remotive_jobs(payload: dict[str, Any] | list[Any]) -> pd.DataFrame:
    items = _payload_items(payload, "jobs")
    rows = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            _live_row(
                provider_rank=idx,
                title=item.get("title"),
                company=item.get("company_name"),
                location=item.get("candidate_required_location") or "Remote",
                posting_date=item.get("publication_date"),
                link=item.get("url"),
                job_id=item.get("id"),
                source="Remotive",
                snippet=_strip_html(item.get("description")),
            )
        )
    return _frame_from_rows(rows)


def parse_arbeitnow_jobs(payload: dict[str, Any] | list[Any]) -> pd.DataFrame:
    items = _payload_items(payload, "data")
    rows = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            _live_row(
                provider_rank=idx,
                title=item.get("title"),
                company=item.get("company_name"),
                location=item.get("location")
                or ("Remote" if item.get("remote") else ""),
                posting_date=item.get("created_at"),
                link=item.get("url"),
                job_id=item.get("slug"),
                source="Arbeitnow",
                snippet=_strip_html(item.get("description")),
            )
        )
    return _frame_from_rows(rows)


def parse_remoteok_jobs(payload: dict[str, Any] | list[Any]) -> pd.DataFrame:
    items = payload if isinstance(payload, list) else _payload_items(payload, "jobs")
    rows = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not item.get("position"):
            continue
        rows.append(
            _live_row(
                provider_rank=idx,
                title=item.get("position"),
                company=item.get("company"),
                location=item.get("location") or "Remote",
                posting_date=item.get("date") or item.get("epoch"),
                link=item.get("url") or item.get("apply_url"),
                job_id=item.get("id"),
                source="RemoteOK",
                snippet=_strip_html(
                    item.get("description") or " ".join(item.get("tags") or [])
                ),
            )
        )
    return _frame_from_rows(rows)


def _payload_items(payload: dict[str, Any] | list[Any], key: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    items = payload.get(key, [])
    return items if isinstance(items, list) else []


def _live_row(
    *,
    provider_rank: int,
    title: Any,
    company: Any,
    location: Any,
    posting_date: Any,
    link: Any,
    job_id: Any,
    source: str,
    snippet: Any = "",
) -> dict[str, Any] | None:
    title_text = _compact_text(title)
    link_text = _compact_text(link)
    if not title_text or not _is_http_url(link_text):
        return None
    return {
        "provider_rank": int(provider_rank),
        "title": title_text,
        "company_name": _compact_text(company),
        "location": _compact_text(location),
        "posting_date": _compact_text(posting_date),
        "job_link": link_text,
        "job_id": _compact_text(job_id),
        "company_profile": "",
        "snippet": _strip_html(snippet),
        "source": source,
    }


def _frame_from_rows(rows: list[dict[str, Any] | None]) -> pd.DataFrame:
    usable = [row for row in rows if row]
    if not usable:
        return empty_live_jobs_frame()
    frame = pd.DataFrame(usable)
    frame = frame.drop_duplicates(subset=["job_link"], keep="first")
    has_job_id = frame["job_id"].astype(str).str.strip().ne("")
    with_ids = frame.loc[has_job_id].drop_duplicates(
        subset=["source", "job_id"], keep="first"
    )
    frame = pd.concat([with_ids, frame.loc[~has_job_id]], ignore_index=True)
    return _ensure_live_columns(frame.reset_index(drop=True))


def _local_filter_provider_frame(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    query_tokens = _token_set(query) - _LIVE_QUERY_STOPWORDS
    if not query_tokens:
        return frame

    keep: list[bool] = []
    for _, row in frame.iterrows():
        text = " ".join(
            str(row.get(field) or "")
            for field in ("title", "company_name", "location", "snippet")
        )
        tokens = _token_set(text)
        keep.append(bool(query_tokens & tokens))

    filtered = frame.loc[keep].reset_index(drop=True)
    if filtered.empty:
        filtered = empty_live_jobs_frame(
            f"{frame.iloc[0].get('source', 'Provider')} returned no rows matching the query terms."
        )
    return filtered


def _strip_html(value: Any, *, max_chars: int = 700) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return _compact_text(text)[:max_chars]


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _append_phrase(parts: list[str], value: Any) -> None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return
    normalized = text.lower()
    if normalized not in {part.lower() for part in parts}:
        parts.append(text)


def _cap_query(query: str, *, max_chars: int) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    if len(query) <= max_chars:
        return query
    tokens = query.split()
    kept: list[str] = []
    for token in tokens:
        candidate = " ".join([*kept, token])
        if len(candidate) > max_chars:
            break
        kept.append(token)
    return " ".join(kept)


def _is_linkedin_job_url(link: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(link)
    except ValueError:
        return False
    host = parsed.netloc.lower()
    return (
        parsed.scheme == "https"
        and (host == "linkedin.com" or host.endswith(".linkedin.com"))
        and "/jobs/view" in parsed.path
    )


def _is_http_url(link: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(link)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _ensure_live_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in empty_live_jobs_frame().columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _live_embedding_text(row: pd.Series) -> str:
    fields = ("title", "company_name", "location", "posting_date", "snippet")
    return " ".join(str(row.get(field) or "").strip() for field in fields).strip()


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = query.astype(np.float32, copy=False)
    matrix = matrix.astype(np.float32, copy=False)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    query_norm[query_norm < 1e-12] = 1.0
    matrix_norm[matrix_norm < 1e-12] = 1.0
    normalized_query = query / query_norm
    normalized_matrix = matrix / matrix_norm
    return (normalized_matrix @ normalized_query.T).reshape(-1)


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9+#.]+", str(text).lower())
        if len(token) >= 3
    }


def _token_overlap(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    return min(1.0, len(query_tokens & text_tokens) / min(10, len(query_tokens)))


def _parse_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        timestamp = int(text)
        if timestamp > 10_000_000_000:
            timestamp //= 1000
        try:
            return datetime.utcfromtimestamp(timestamp).date()
        except (OverflowError, OSError, ValueError):
            return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None
