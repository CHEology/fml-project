"""
app/components/resume_upload.py

Reusable resume input widget supporting PDF upload, plain text upload,
URL import, and direct text paste. Centralizes all resume ingestion logic
so individual pages don't duplicate it.

Owner: @trp8625
"""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import streamlit as st


def extract_uploaded_text(uploaded_file) -> str:
    """
    Extract plain text from an uploaded PDF or TXT file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Extracted text string, or empty string if extraction fails.
    """
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            st.warning(
                "pdfplumber is required for PDF parsing. Install it with: pip install pdfplumber"
            )
            return ""

        pages: list[str] = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                if not text.strip():
                    text = page.extract_text() or ""
                pages.append(text)
        return _clean_extracted_pdf_text("\n".join(pages))

    return ""


def _clean_extracted_pdf_text(text: str) -> str:
    """Remove PDF encoding artifacts and normalize whitespace."""
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_public_webpage_text(url: str) -> tuple[str, str]:
    """
    Fetch and extract visible text from a public URL.
    LinkedIn URLs are explicitly blocked.

    Args:
        url: Public http/https URL to import.

    Returns:
        Tuple of (extracted text capped at 8000 chars, domain string).

    Raises:
        ValueError: If the URL is invalid, private, or returns too little text.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid public http or https URL.")
    if "linkedin.com" in parsed.netloc.lower():
        raise ValueError(
            "LinkedIn pages are not imported here. Paste profile text directly."
        )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=12) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            raise ValueError("That URL did not return an HTML page.")
        html = response.read(1_500_000).decode("utf-8", errors="ignore")

    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    html = re.sub(
        r"(?i)</?(p|div|section|article|li|h1|h2|h3|h4|h5|h6|br)[^>]*>", "\n", html
    )
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    text = unescape(html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    cleaned = text.strip()
    if len(cleaned) < 120:
        raise ValueError(
            "The page did not expose enough public text to use as a resume input."
        )
    return cleaned[:8000], parsed.netloc.lower()


def resume_input_widget(key_prefix: str = "") -> None:
    """
    Render the full resume input panel: file uploader, URL import,
    text area, and word count caption. Reads and writes to
    st.session_state.resume_text and st.session_state.resume_source.

    Args:
        key_prefix: Optional prefix for Streamlit widget keys to avoid
                    collisions when the widget is rendered on multiple pages.
    """
    # File upload
    uploaded = st.file_uploader(
        "Drag and drop resume here (PDF or TXT)",
        type=["pdf", "txt"],
        key=f"{key_prefix}uploader",
    )
    if uploaded is not None:
        parsed = extract_uploaded_text(uploaded)
        if parsed:
            st.session_state.resume_text = parsed
            st.session_state.resume_source = f"Uploaded file: {uploaded.name}"
        else:
            st.warning(
                "Could not extract text from the uploaded file. Paste the resume text below instead."
            )

    # URL import
    url_col, import_col = st.columns([0.76, 0.24], gap="small")
    with url_col:
        url_val = st.text_input(
            "Public profile or portfolio URL",
            value=st.session_state.get("public_profile_url", ""),
            placeholder="https://portfolio.example.com/about",
            label_visibility="collapsed",
            key=f"{key_prefix}url_input",
        )
        st.session_state.public_profile_url = url_val
    with import_col:
        import_clicked = st.button("Import page", key=f"{key_prefix}import_btn", width="stretch")

    if import_clicked:
        try:
            with st.spinner("Importing public page text..."):
                imported_text, imported_host = fetch_public_webpage_text(
                    st.session_state.public_profile_url
                )
            st.session_state.resume_text = imported_text
            st.session_state.resume_source = f"Imported public webpage: {imported_host}"
            st.rerun()
        except ValueError as exc:
            st.warning(str(exc))
        except Exception:
            st.warning(
                "Could not import that page. Try another public URL or paste the resume text directly."
            )

    st.caption(
        "Public webpage import is for generic portfolio or resume pages. "
        "LinkedIn pages are not imported here."
    )

    # Text area
    st.session_state.resume_text = st.text_area(
        "Paste resume text",
        value=st.session_state.get("resume_text", ""),
        height=340,
        placeholder="Paste a resume, portfolio bio, or achievement summary here...",
        key=f"{key_prefix}text_area",
    )

    word_count = len(st.session_state.resume_text.split())
    st.caption(f"{word_count} words · click Analyze to evaluate.")
