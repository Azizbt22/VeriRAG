
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(
    model_name: str = "llama3.1:8b",
    temperature: float = 0.2,
) -> BaseChatModel:
    """
    Factory function returning a LangChain-compatible LLM.

    This abstraction allows easy swapping between local models
    (Ollama) and API-based models later without changing the agent.
    """
    return ChatOllama(
        model=model_name,
        temperature=temperature,
    )
