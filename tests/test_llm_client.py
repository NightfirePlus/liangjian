"""会话级模型配置和 OpenAI 兼容请求的回归测试。"""
from __future__ import annotations

import pytest

from core.nlp.llm_client import LLMError, complete_json
from core.nlp.parser import parse_user_input


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}],
        }


def test_complete_json_uses_explicit_session_settings(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return _Response()

    monkeypatch.setattr("core.nlp.llm_client.requests.post", fake_post)
    result = complete_json(
        "system", "user",
        api_key="session-secret",
        base_url="https://example.test/compatible/v1/",
        model="session-model",
    )

    assert result == {"ok": True}
    assert captured["url"] == "https://example.test/compatible/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer session-secret"
    assert captured["payload"]["model"] == "session-model"


def test_complete_json_rejects_incomplete_explicit_settings():
    with pytest.raises(LLMError, match="配置不完整"):
        complete_json("system", "user", api_key="", base_url="", model="")


def test_parser_accepts_injected_session_client():
    calls = []

    def complete(system_prompt: str, user_prompt: str) -> dict:
        calls.append((system_prompt, user_prompt))
        return {
            "template": "ma_cross",
            "symbols": ["510300"],
            "params": {"fast": 5, "slow": 20},
            "start": "2023-01-01",
            "end": "2024-01-01",
            "missing": [],
            "ambiguities": [],
        }

    parsed = parse_user_input("测试双均线", use_llm=True, complete=complete)

    assert calls
    assert not parsed.needs_clarification
    assert parsed.strategy_config.template == "ma_cross"
    assert not parsed.degraded
