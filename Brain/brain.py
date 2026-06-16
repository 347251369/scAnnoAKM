import re
import time
import requests
from typing import List, Dict, Any

_KEY_RE = re.compile(r"^(?:[^:]+:)?(.+)$")
class LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str, timeout: int = 90):
        self.api_base = api_base.rstrip("/")
        raw_key = (api_key or "").strip()

        m = _KEY_RE.search(raw_key)
        if m:
            raw_key = m.group(1)
        try:
            raw_key.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError("QWEN_API_KEY need all ASCII Characters")
        if not raw_key.startswith("sk-"):
            raise ValueError("QWEN_API_KEY need begin with'sk-'")

        self.api_key = raw_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], extra_body: Dict[str, Any] = None) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {"model": self.model, "messages": messages}
        if extra_body:
            payload.update(extra_body)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"[LLMClient] HTTP {r.status_code} {r.reason}: {r.text}")
            data = r.json()
        except requests.RequestException as e:
            raise RuntimeError(f"[LLMClient] request failed: {e}")
        return data["choices"][0]["message"]["content"]