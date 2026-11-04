from core.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    GEMINI_MODEL,
    GITHUB_MODEL,
    GITHUB_TOKEN,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MAX_TOKENS,
)


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def make_llm():
    if LLM_PROVIDER == "claude":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
        )
    if LLM_PROVIDER == "github":
        from langchain_openai import ChatOpenAI

        model_name = (
            GITHUB_MODEL.split("/")[-1]
            if "/" in (GITHUB_MODEL or "")
            else GITHUB_MODEL
        )
        return ChatOpenAI(
            api_key=GITHUB_TOKEN,
            base_url="https://models.inference.ai.azure.com",
            model=model_name,
            max_tokens=MAX_TOKENS,
        )
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            max_tokens=MAX_TOKENS,
        )
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            api_key=GOOGLE_API_KEY,
            model=GEMINI_MODEL,
            max_output_tokens=MAX_TOKENS,
        )
    raise ValueError(
        "LLM_PROVIDER must be one of: claude, github, groq, gemini"
    )
