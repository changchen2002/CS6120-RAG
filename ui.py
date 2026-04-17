import html
import os
import json

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
RAG_API_KEY = os.getenv("RAG_API_KEY", "").strip()

_SOURCES_PREVIEW_MAX_WORDS = 300


def _api_headers(**extra: str) -> dict:
    h = {**extra}
    if RAG_API_KEY:
        h["X-API-Key"] = RAG_API_KEY
    return h


def _preview_passage_words(text: str, max_words: int = _SOURCES_PREVIEW_MAX_WORDS) -> str:
    if not text or not str(text).strip():
        return ""
    words = str(text).split()
    if len(words) <= max_words:
        return str(text).strip()
    return " ".join(words[:max_words]) + " …"


def _render_assistant_extras(retrieval_info, sources):
    if retrieval_info:
        conf = retrieval_info.get("confidence", "?")
        mx = retrieval_info.get("max_similarity")
        mn = retrieval_info.get("mean_similarity")
        note = retrieval_info.get("instructor_note", "")
        if conf == "low":
            st.warning(
                f"**Retrieval confidence: {conf}** (max ≈ {mx}, mean ≈ {mn}). {note}"
            )
        else:
            st.info(
                f"**Retrieval confidence: {conf}** — max ≈ {mx}, mean ≈ {mn}. {note}"
            )
    if sources:
        st.markdown("**Sources** (verify claims against these)")
        for i, s in enumerate(sources):
            title = s.get("title", "Untitled")
            url = (s.get("url") or "").strip()
            passage = (s.get("passage") or "").strip()
            sim = s.get("similarity_score")
            idx = i + 1
            if sim is not None:
                st.caption(f"Cosine similarity: {sim}")
            if url:
                safe_href = html.escape(url, quote=True)
                safe_title = html.escape(title)
                st.markdown(
                    f'<p>[{idx}] <a href="{safe_href}" target="_blank" rel="noopener noreferrer">{safe_title}</a></p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**[{idx}]** {html.escape(title)}")
            if passage:
                preview = _preview_passage_words(passage)
                with st.expander(f"Passage preview [{idx}] (≤{_SOURCES_PREVIEW_MAX_WORDS} words)"):
                    st.text(preview)


st.set_page_config(page_title="Citation RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Citation RAG")
st.caption("Answers use retrieved arXiv abstracts from your local Qdrant index. All processing stays on your machine.")

with st.sidebar:
    st.header("Session")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_assistant_extras(msg.get("retrieval"), msg.get("sources") or [])

if prompt := st.chat_input("Ask a question about the indexed papers…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    full_text = ""
    sources_out = None
    retrieval_out = None
    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.empty()
        status.info("Embedding → Qdrant → Ollama… (first tokens can be slow on CPU)")
        try:
            with requests.post(
                f"{API_URL}/query_stream",
                json={"question": prompt},
                stream=True,
                headers=_api_headers(Accept="text/plain"),
                timeout=600,
            ) as response:
                if response.status_code == 401:
                    full_text = (
                        "API returned 401: set RAG_API_KEY for both services (see README)."
                    )
                    placeholder.error(full_text)
                elif response.status_code != 200:
                    full_text = f"HTTP {response.status_code}"
                    placeholder.error(full_text)
                else:
                    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if "chunk" in data:
                            status.empty()
                            full_text += data["chunk"]
                            placeholder.markdown(full_text + "▌")
                        if "sources" in data:
                            sources_out = data["sources"]
                        if "retrieval" in data:
                            retrieval_out = data["retrieval"]
                        if "error" in data:
                            status.empty()
                            full_text = data["error"]
                            placeholder.error(full_text)
                            break
                        if data.get("done"):
                            if "sources" in data:
                                sources_out = data["sources"]
                            if "retrieval" in data:
                                retrieval_out = data["retrieval"]
                            break
                    if full_text and not full_text.startswith("⚠️") and "HTTP" not in full_text:
                        placeholder.markdown(full_text)
                    status.empty()
        except Exception as e:
            full_text = f"Request failed: {e}"
            placeholder.error(full_text)
            status.empty()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "sources": sources_out or [],
            "retrieval": retrieval_out,
        }
    )

    st.rerun()
