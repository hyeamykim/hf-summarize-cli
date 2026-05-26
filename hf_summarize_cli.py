import argparse
import json
import os
import re
import time
from html.parser import HTMLParser
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-de-en"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
HF_API_URL = "https://router.huggingface.co/hf-inference/models/{model}"
MAX_INPUT_CHARS = 2048

class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Please provide a valid http(s) URL.")
    return url


def fetch_webpage_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "hf-summarize-cli/1.0",
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to fetch URL: {exc}") from exc

    parser = _VisibleTextParser()
    parser.feed(html)
    text = re.sub(r"\s+", " ", parser.get_text()).strip()
    if not text:
        raise RuntimeError("No readable text found on the webpage.")
    return text[:MAX_INPUT_CHARS]


def _hf_request(model: str, text: str, token: Optional[str]) -> dict:
    payload = json.dumps({"inputs": text}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(HF_API_URL.format(model=model), data=payload, headers=headers, method="POST")

    for _ in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Hugging Face API request failed for {model}: {body or exc}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Hugging Face API for {model}: {exc}") from exc

        if isinstance(data, dict) and "estimated_time" in data:
            time.sleep(min(float(data["estimated_time"]), 5.0))
            continue
        return data

    raise RuntimeError(f"Model {model} is still loading. Please retry shortly.")


def summarize_url(url: str, token: Optional[str] = None) -> str:
    _validate_url(url)
    source_text = fetch_webpage_text(url)

    translation_response = _hf_request(TRANSLATION_MODEL, source_text, token)
    if not isinstance(translation_response, list) or "translation_text" not in translation_response[0]:
        raise RuntimeError("Unexpected translation response from Hugging Face API.")

    english_text = translation_response[0]["translation_text"]

    summary_response = _hf_request(SUMMARIZATION_MODEL, english_text, token)
    if not isinstance(summary_response, list) or "summary_text" not in summary_response[0]:
        raise RuntimeError("Unexpected summarization response from Hugging Face API.")

    summary = summary_response[0]["summary_text"].strip()
    if not summary:
        raise RuntimeError("Received an empty summary from Hugging Face API.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a German webpage URL into an English paragraph using Hugging Face models."
    )
    parser.add_argument("url", help="German webpage URL to summarize")
    args = parser.parse_args()

    token = os.getenv("HF_API_TOKEN")

    try:
        print(summarize_url(args.url, token=token))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
