import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import core.llm


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-5"])
def test_make_llm_configures_claude_model_and_max_tokens(monkeypatch, model):
    chat_anthropic = MagicMock()
    monkeypatch.setattr(core.llm, "LLM_PROVIDER", "claude")
    monkeypatch.setattr(core.llm, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(core.llm, "CLAUDE_MODEL", model)
    monkeypatch.setattr(core.llm, "MAX_TOKENS", 2048)

    with patch.dict(
        sys.modules,
        {"langchain_anthropic": SimpleNamespace(ChatAnthropic=chat_anthropic)},
    ):
        core.llm.make_llm()

    chat_anthropic.assert_called_once_with(
        api_key="test-key",
        model=model,
        max_tokens=2048,
    )


def test_make_llm_requires_anthropic_api_key(monkeypatch):
    monkeypatch.setattr(core.llm, "LLM_PROVIDER", "claude")
    monkeypatch.setattr(core.llm, "ANTHROPIC_API_KEY", None)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        core.llm.make_llm()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('[{"id": 1}]', '[{"id": 1}]'),
        ([{"type": "text", "text": '[{"id": 1}]'}], '[{"id": 1}]'),
        (["[", {"type": "text", "text": '{"id": 1}]'}], '[{"id": 1}]'),
        ([{"type": "thinking", "thinking": "private"},
          {"type": "tool_use", "name": "inspect", "input": {}},
          {"type": "text", "text": "answer"}], "answer"),
        ([], ""),
        (None, ""),
    ],
)
def test_message_text_handles_claude_content_blocks(content, expected):
    assert core.llm.message_text(SimpleNamespace(content=content)) == expected


@pytest.mark.parametrize("blocked", [False, True])
def test_extract_analysis_accepts_claude_content(blocked):
    import json
    from agents.reporter import _extract_analysis

    analysis = {"direct_answer": {"answer": "4"}, "charts": []}
    text = json.dumps(analysis)
    content = [{"type": "text", "text": text}] if blocked else text
    assert _extract_analysis({"messages": [SimpleNamespace(content=content)]}) == analysis


@pytest.mark.parametrize(
    "provider, module_name, class_name, model_setting, key_setting, token_setting",
    [
        ("github", "langchain_openai", "ChatOpenAI", "GITHUB_MODEL", "GITHUB_TOKEN", "max_tokens"),
        ("groq", "langchain_groq", "ChatGroq", "GROQ_MODEL", "GROQ_API_KEY", "max_tokens"),
        ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI", "GEMINI_MODEL", "GOOGLE_API_KEY", "max_output_tokens"),
    ],
)
def test_other_providers_remain_supported(
    monkeypatch, provider, module_name, class_name, model_setting, key_setting, token_setting
):
    constructor = MagicMock()
    monkeypatch.setattr(core.llm, "LLM_PROVIDER", provider)
    monkeypatch.setattr(core.llm, model_setting, "test-model")
    monkeypatch.setattr(core.llm, key_setting, "test-key")
    monkeypatch.setattr(core.llm, "MAX_TOKENS", 128)
    with patch.dict(sys.modules, {module_name: SimpleNamespace(**{class_name: constructor})}):
        assert core.llm.make_llm() is constructor.return_value
    expected = {"api_key": "test-key", "model": "test-model", token_setting: 128}
    if provider == "github":
        expected["base_url"] = "https://models.inference.ai.azure.com"
    constructor.assert_called_once_with(**expected)


def test_unsupported_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(core.llm, "LLM_PROVIDER", "unsupported")
    with pytest.raises(ValueError, match="LLM_PROVIDER must be one of"):
        core.llm.make_llm()
