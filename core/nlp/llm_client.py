"""OpenAI 兼容接口的最小封装。

不引入 SDK 依赖，直接用 requests 调 {LLM_BASE_URL}/chat/completions。
config.py 的三个环境变量（LLM_API_KEY/LLM_BASE_URL/LLM_MODEL）均非空时
LLM_AVAILABLE 为 True，由调用方（parser.py）决定是否走降级路径。
"""
from __future__ import annotations

import json

import requests

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS



class LLMError(Exception):
    """大模型调用失败或返回内容不是合法 JSON，统一包成这个异常供上层捕获降级。"""


def complete_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """调用大模型，要求其只返回一个 JSON object，解析后以 dict 形式返回。

    失败（网络错误、HTTP 非 2xx、返回内容无法解析为 JSON）统一抛 LLMError。
    """
    effective_key = LLM_API_KEY if api_key is None else api_key
    effective_base_url = LLM_BASE_URL if base_url is None else base_url
    effective_model = model or LLM_MODEL
    if not effective_key or not effective_base_url or not effective_model:
        raise LLMError("大模型配置不完整，请填写 API Key、接口 URL 和模型名")
    url = effective_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {effective_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": effective_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except requests.RequestException as err:
        raise LLMError(f"调用大模型接口失败: {err}") from err
    except (KeyError, IndexError, ValueError) as err:
        raise LLMError(f"大模型返回结构不符合预期: {err}") from err

    content = _strip_code_fence(content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as err:
        raise LLMError(f"大模型返回内容不是合法 JSON: {err}") from err

    if not isinstance(parsed, dict):
        raise LLMError("大模型返回内容不是 JSON object")
    return parsed


def _strip_code_fence(text: str) -> str:
    """大模型有时会把 JSON 包在 ```json ... ``` 代码块里，这里去掉围栏。"""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s
