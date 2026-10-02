"""번역 로직: OpenAI API 호출, 프롬프트, 오류 분류."""

import os
import time
from collections.abc import Iterator
from functools import lru_cache

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-6-astra"
MAX_CHARS = 5000
RATE_LIMIT_RETRY_DELAY = 2  # 초
# 5,000자 번역은 언어 하나에 40초 이상 걸릴 수 있다. 스트리밍에서는 조각 사이의 대기 시간에 적용된다.
REQUEST_TIMEOUT = 120  # 초
# 언어 감지는 앞부분만 보내 비용을 줄인다.
DETECT_SAMPLE_CHARS = 500

SYSTEM_PROMPT = """You are a professional translator. Translate the user's text into {target_language}.
- Preserve meaning, tone, line breaks and formatting.
- Keep URLs, code, and email addresses unchanged.
- Write proper nouns (people, companies, products, places) in their established {target_language} form. If there is none, keep the original.
- Output ONLY the translated text, without explanations."""

DETECT_PROMPT = """Identify the language of the user's text.
Reply with ONLY its ISO 639-1 code in lowercase (for example: ko, en, ja, zh). No other words."""

# 원문 언어 감지 결과(ISO 639-1)를 한국어 이름으로 표시할 때 사용
LANGUAGE_NAMES = {
    "ko": "한국어",
    "en": "영어",
    "ja": "일본어",
    "zh": "중국어",
    "es": "스페인어",
    "fr": "프랑스어",
    "de": "독일어",
    "vi": "베트남어",
    "ru": "러시아어",
}


class TranslationError(Exception):
    """사용자에게 그대로 보여줄 한국어 메시지를 담은 오류."""


def _get_setting(name: str, default: str | None = None) -> str | None:
    """st.secrets → os.environ(.env 포함) 순서로 설정값을 읽는다."""
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # secrets.toml이 없거나 Streamlit 밖에서 실행된 경우
        pass
    return os.environ.get(name) or default


def get_api_key() -> str | None:
    return _get_setting("OPENAI_API_KEY")


def get_model() -> str:
    return _get_setting("OPENAI_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=4)
def _client(api_key: str) -> OpenAI:
    # 429 재시도는 _create에서 직접 1회만 수행한다.
    return OpenAI(api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT)


def _is_quota_error(e: openai.RateLimitError) -> bool:
    # 사용 한도(크레딧) 소진도 429로 오지만, 재시도해도 소용없고 안내가 달라야 한다.
    return getattr(e, "code", None) == "insufficient_quota"


def _create(model: str, messages: list[dict], stream: bool):
    client = _client(get_api_key())
    try:
        return client.chat.completions.create(model=model, messages=messages, stream=stream)
    except openai.RateLimitError as e:
        if _is_quota_error(e):
            raise
        time.sleep(RATE_LIMIT_RETRY_DELAY)
        return client.chat.completions.create(model=model, messages=messages, stream=stream)


def _to_translation_error(e: openai.APIError, model: str) -> TranslationError:
    if isinstance(e, openai.AuthenticationError):
        return TranslationError("API 키가 올바르지 않습니다.")
    if isinstance(e, openai.NotFoundError):
        return TranslationError(f"모델({model})을 사용할 수 없습니다. OPENAI_MODEL 설정을 확인하세요.")
    if isinstance(e, openai.RateLimitError):
        if _is_quota_error(e):
            return TranslationError("API 사용 한도를 초과했습니다. OpenAI 계정의 결제·사용량 설정을 확인하세요.")
        return TranslationError("요청이 많습니다. 잠시 후 다시 시도해 주세요.")
    # APITimeoutError는 APIConnectionError의 하위 클래스라 먼저 확인한다.
    if isinstance(e, openai.APITimeoutError):
        return TranslationError(
            "번역 응답 시간이 초과되었습니다. 글을 나누어 다시 시도해 주세요."
        )
    if isinstance(e, openai.APIConnectionError):
        return TranslationError(
            "번역 중 오류가 발생했습니다: 서버에 연결할 수 없습니다. 네트워크를 확인하세요."
        )
    return TranslationError(f"번역 중 오류가 발생했습니다: {e.message}")


def translate_stream(text: str, target_language: str, model: str) -> Iterator[str]:
    """text를 target_language(예: "English")로 번역하며 결과 조각을 차례로 내보낸다.

    실패 시 TranslationError.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(target_language=target_language)},
        {"role": "user", "content": text},
    ]
    try:
        for chunk in _create(model, messages, stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except openai.APIError as e:
        raise _to_translation_error(e, model) from None


def detect_language(text: str, model: str) -> str:
    """원문 언어의 ISO 639-1 코드(예: "ko")를 반환한다. 실패 시 TranslationError."""
    messages = [
        {"role": "system", "content": DETECT_PROMPT},
        {"role": "user", "content": text[:DETECT_SAMPLE_CHARS]},
    ]
    try:
        response = _create(model, messages, stream=False)
    except openai.APIError as e:
        raise _to_translation_error(e, model) from None
    code = (response.choices[0].message.content or "").lower().strip(" .\"'`")
    # "zh-cn" 같은 응답도 앞 두 글자로 정규화
    return code.split("-")[0][:2]
