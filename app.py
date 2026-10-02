import queue
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from translator import (
    LANGUAGE_NAMES,
    MAX_CHARS,
    TranslationError,
    detect_language,
    get_api_key,
    get_model,
    translate_stream,
)

# 언어 코드 → (화면 표시 이름, 결과 탭 이름, 프롬프트에 넣을 언어 이름).
# 중국어는 간체/번체 선택에 따라 CHINESE_VARIANTS 값으로 정해진다.
TARGET_LANGUAGES = {
    "en": ("영어", "English", "English"),
    "ja": ("일본어", "日本語", "Japanese"),
    "zh": ("중국어", None, None),
}
CHINESE_VARIANTS = {
    "간체": ("中文(简体)", "Simplified Chinese"),
    "번체": ("中文(繁體)", "Traditional Chinese"),
}
CACHE_MAX_ENTRIES = 500
STREAM_REDRAW_INTERVAL = 0.1  # 초. 스트리밍 중 화면 갱신 간격

CUSTOM_CSS = """
<style>
/* .streamlit/config.toml의 font/codeFont가 가리키는 웹폰트 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Noto+Sans+JP:wght@400;500;700&family=Noto+Sans+SC:wght@400;500;700&family=Noto+Sans+TC:wght@400;500;700&display=swap');
/* 제목 옆 앵커 링크 아이콘 숨김 */
[data-testid="stHeaderActionElements"] { display: none; }
/* 결과 탭 글자를 조금 크게 */
[data-testid="stTabs"] button[role="tab"] p { font-size: 1rem; font-weight: 500; }
/* 번역 결과 블록: 여백을 넓혀 읽기 편하게 */
[data-testid="stTabs"] [data-testid="stCode"] pre { padding: 1rem 1.25rem; line-height: 1.7; }
/* 글자 수 표시. 한도를 넘으면 빨간색 */
.char-counter { text-align: right; font-size: 0.85rem; margin-top: -0.75rem; opacity: 0.7; }
.char-counter.over { color: #E5484D; font-weight: 600; opacity: 1; }
</style>
"""

# st.text_area는 입력창을 벗어나야 값을 서버로 보내므로, 입력 중 글자 수는 브라우저에서 갱신한다.
# 글자 수는 Python의 len()과 같도록 코드 포인트 단위로 센다.
# 버튼 아이콘(Material 아이콘 글자)이 화면 낭독기에 "translate"처럼 읽히지 않게 aria-hidden도 붙인다.
PAGE_SCRIPT = f"""
<script>
(() => {{
  if (window.__pageScriptBound) return;
  window.__pageScriptBound = true;
  const hideIcons = () => document
    .querySelectorAll('button [data-testid="stIconMaterial"]:not([aria-hidden])')
    .forEach((icon) => icon.setAttribute("aria-hidden", "true"));
  hideIcons();
  new MutationObserver(hideIcons).observe(document.body, {{ childList: true, subtree: true }});
  document.addEventListener("input", (e) => {{
    const ta = e.target;
    if (!(ta instanceof HTMLTextAreaElement) || ta.getAttribute("aria-label") !== "번역할 글") return;
    const el = document.querySelector(".char-counter");
    if (!el) return;
    const n = [...ta.value].length;
    el.textContent = n.toLocaleString("en-US") + " / {MAX_CHARS:,}자";
    el.classList.toggle("over", n > {MAX_CHARS});
  }});
}})();
</script>
"""


@st.cache_resource
def _result_cache() -> tuple[OrderedDict, threading.Lock]:
    """번역·감지 결과를 모든 세션이 함께 쓰는 LRU 캐시. 성공한 결과만 넣는다."""
    return OrderedDict(), threading.Lock()


def cache_get(key: tuple):
    cache, lock = _result_cache()
    with lock:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]


def cache_put(key: tuple, value: str) -> None:
    cache, lock = _result_cache()
    with lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > CACHE_MAX_ENTRIES:
            cache.popitem(last=False)


def stream_translations(text: str, model: str, jobs: dict, tab_labels: dict) -> tuple[dict, str | None]:
    """jobs({언어 코드: 프롬프트 언어 이름})를 병렬로 번역하며 탭에 실시간으로 표시한다.

    ({코드: {"text": ...} 또는 {"error": ...}}, 감지된 원문 언어 코드)를 반환한다.
    작업 스레드는 Streamlit을 건드리지 않고 큐로만 결과를 보내며, 화면 갱신은 이 함수(메인 스레드)가 한다.
    """
    tabs = st.tabs([tab_labels[code] for code in jobs])
    placeholders = {code: tab.empty() for code, tab in zip(jobs, tabs)}
    show = lambda code, body: placeholders[code].code(body, language=None, wrap_lines=True)

    events: queue.Queue = queue.Queue()

    def work_translate(code: str, language: str) -> None:
        try:
            for piece in translate_stream(text, language, model):
                events.put((code, "chunk", piece))
            events.put((code, "done", None))
        except TranslationError as e:
            events.put((code, "error", str(e)))
        except Exception as e:
            events.put((code, "error", f"번역 중 오류가 발생했습니다: {e}"))

    def work_detect() -> None:
        try:
            events.put(("_detect", "done", detect_language(text, model)))
        except Exception:
            # 감지 실패는 번역에 영향을 주지 않으므로 표시만 생략한다.
            events.put(("_detect", "error", None))

    outcomes, pending = {}, set()
    pool = ThreadPoolExecutor(max_workers=len(jobs) + 1)
    try:
        for code, language in jobs.items():
            cached = cache_get(("translate", text, language, model))
            if cached is not None:
                outcomes[code] = {"text": cached}
                show(code, cached)
            else:
                pending.add(code)
                show(code, "…")
                pool.submit(work_translate, code, language)

        detected = cache_get(("detect", text, model))
        if detected is None:
            pending.add("_detect")
            pool.submit(work_detect)

        buffers, last_drawn = defaultdict(str), {}
        while pending:
            code, kind, value = events.get()
            if code == "_detect":
                pending.discard(code)
                if kind == "done":
                    detected = value
                    cache_put(("detect", text, model), value)
                continue
            if kind == "chunk":
                buffers[code] += value
                now = time.monotonic()
                if now - last_drawn.get(code, 0) >= STREAM_REDRAW_INTERVAL:
                    show(code, buffers[code] + " ▌")
                    last_drawn[code] = now
                continue

            pending.discard(code)
            translated = buffers[code].strip()
            if kind == "error":
                outcomes[code] = {"error": value}
                placeholders[code].error(value)
            elif not translated:
                outcomes[code] = {"error": "번역 중 오류가 발생했습니다: 빈 응답을 받았습니다."}
                placeholders[code].error(outcomes[code]["error"])
            else:
                outcomes[code] = {"text": translated}
                cache_put(("translate", text, jobs[code], model), translated)
                show(code, translated)
    finally:
        # 사용자가 도중에 다시 실행하면 남은 작업은 버리고 기다리지 않는다.
        pool.shutdown(wait=False)

    return {code: outcomes[code] for code in jobs}, detected


def build_download(result: dict) -> str:
    parts = [f"[원문]\n{result['source']}"]
    for code, outcome in result["translations"].items():
        label = result["labels"][code]
        parts.append(f"[{label}]\n{outcome.get('text') or '(번역 실패: ' + outcome['error'] + ')'}")
    return "\n\n".join(parts) + "\n"


def render_result(result: dict, current_text: str) -> None:
    if result["source"] != current_text:
        st.warning("입력한 글이 바뀌었습니다. 아래는 이전 글의 번역입니다. 새로 번역하려면 **번역하기**를 누르세요.")

    detected = result["detected"]
    if detected:
        st.caption(f"감지된 원문 언어: **{LANGUAGE_NAMES.get(detected, detected)}**")

    codes = list(result["translations"])
    for code, tab in zip(codes, st.tabs([result["tab_labels"][c] for c in codes])):
        with tab:
            if detected == code:
                if code == "zh":
                    st.info(f"원문이 중국어여서 {result['chinese_variant']} 표기로 바꾼 결과입니다.")
                else:
                    st.info(f"원문이 이미 {TARGET_LANGUAGES[code][0]}입니다.")
            outcome = result["translations"][code]
            if "error" in outcome:
                st.error(outcome["error"])
            else:
                st.code(outcome["text"], language=None, wrap_lines=True)

    st.download_button(
        "전체 결과 다운로드 (.txt)",
        data=build_download(result),
        file_name="translation.txt",
        mime="text/plain",
        icon=":material/download:",
        width="stretch",
        on_click="ignore",
    )


st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.html(PAGE_SCRIPT, unsafe_allow_javascript=True)

st.title("🌐 다국어 번역기")
st.caption("입력한 글을 영어 · 일본어 · 중국어로 한 번에 번역합니다.")

if not get_api_key():
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

text = st.text_area(
    "번역할 글",
    height=200,
    key="source_text",
    placeholder="번역할 글을 입력하거나 붙여넣으세요.",
)
over_class = " over" if len(text) > MAX_CHARS else ""
st.markdown(
    f"<div class='char-counter{over_class}'>{len(text):,} / {MAX_CHARS:,}자</div>",
    unsafe_allow_html=True,
)

lang_col, variant_col = st.columns([2, 1])
with lang_col:
    selected = st.multiselect(
        "대상 언어",
        options=list(TARGET_LANGUAGES),
        default=list(TARGET_LANGUAGES),
        format_func=lambda code: TARGET_LANGUAGES[code][0],
        placeholder="언어를 선택하세요",
        key="target_codes",
    )
with variant_col:
    chinese_variant = st.radio(
        "중국어 표기", list(CHINESE_VARIANTS), horizontal=True, key="chinese_variant"
    )

start_translation = False
if st.button("번역하기", type="primary", icon=":material/translate:", width="stretch"):
    if not text.strip():
        st.warning("번역할 글을 입력해 주세요.")
    elif len(text) > MAX_CHARS:
        st.warning(f"최대 {MAX_CHARS:,}자까지 번역할 수 있습니다.")
    elif not selected:
        st.warning("번역할 언어를 하나 이상 선택해 주세요.")
    else:
        start_translation = True

st.divider()

# 결과 영역 전체를 하나의 슬롯에 그려, 새 번역이 시작되면 이전 결과가 화면에 남지 않게 한다.
results_slot = st.empty()

if start_translation:
    model = get_model()
    labels, tab_labels, jobs = {}, {}, {}
    for code in selected:
        label, tab_label, language = TARGET_LANGUAGES[code]
        if code == "zh":
            tab_label, language = CHINESE_VARIANTS[chinese_variant]
            label = f"중국어({chinese_variant})"
        labels[code] = label
        tab_labels[code] = tab_label
        jobs[code] = language

    with results_slot.container(), st.spinner("번역 중..."):
        outcomes, detected = stream_translations(text, model, jobs, tab_labels)

    st.session_state["result"] = {
        "source": text,
        "detected": detected,
        "chinese_variant": chinese_variant,
        "labels": labels,
        "tab_labels": tab_labels,
        "translations": outcomes,
    }
    # 결과는 session_state에서 다시 그려 안내 문구·다운로드 버튼까지 일관되게 보여준다.
    st.rerun()

# 결과는 session_state에 보관해 다른 위젯을 조작해도 사라지지 않게 한다.
result = st.session_state.get("result")
with results_slot.container():
    if result:
        render_result(result, text)
    else:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center; padding:1.5rem 0; opacity:0.7'>"
                "번역 결과가 여기에 표시됩니다.<br>글을 입력하고 <b>번역하기</b>를 눌러 보세요."
                "</div>",
                unsafe_allow_html=True,
            )
