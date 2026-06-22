from __future__ import annotations

from turtle import config_dict

import ollama

class OllamaChatModel:
    """Ollama  adapter.

    The application layer sees only ChatModel.generate(), so replacing  Ollama
    with OpenAI, vLLM, or a Hugging Face end point means adding another adapter
    """

    def __init__(self, model: str, temperature: float = 0.1) -> None:
        self.model = model
        self.temperature = temperature

    def generate(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": self.temperature},
        }

        if json_mode:
            kwargs["format"] = "json"

        response = ollama.chat(**kwargs)
        return response["message"]["content"]

def build_chat_model(config: dict) -> OllamaChatModel:
    provider = config["provider"]

    if provider == "ollama":
        return OllamaChatModel(
            model=config["model"],
            temperature=float(config.get("temperature", 0.1))
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
