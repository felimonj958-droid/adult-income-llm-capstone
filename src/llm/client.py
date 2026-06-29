import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class LLMClientError(Exception):
    pass


class LLMClient:
    """
    Minimal HTTP client wrapper around an OpenAI-compatible chat API.

    Expects environment variables:
    - LLM_API_BASE_URL: e.g. https://api.tokenfactory.us-central1.nebius.com/v1
    - NEBIUS_API_KEY: your Nebius API key
    - LLM_MODEL_NAME: e.g. Qwen/Qwen3.5-397B-A17B
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_API_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY") or os.getenv("LLM_API_KEY", "")
        self.default_model = default_model or os.getenv("LLM_MODEL_NAME", "")

        if not self.base_url:
            raise LLMClientError("LLM_API_BASE_URL is not set.")
        if not self.api_key:
            raise LLMClientError("NEBIUS_API_KEY is not set.")
        if not self.default_model:
            raise LLMClientError("LLM_MODEL_NAME is not set.")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def parse_json(self, text: str) -> dict:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines.strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"LLM returned invalid JSON: {text}") from exc

    def _coerce_message_text(self, message: Dict[str, Any]) -> str:
        content = message.get("content")
        reasoning = message.get("reasoning")

        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        text_parts.append(str(item["text"]))
                    elif "content" in item and item["content"]:
                        text_parts.append(str(item["content"]))
                elif isinstance(item, str) and item.strip():
                    text_parts.append(item)
            combined = "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()
            if combined:
                return combined

        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        raise LLMClientError(f"LLM API returned an empty message payload: {message}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 250,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMClientError(f"Request to LLM API failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError(f"LLM API did not return valid JSON: {response.text}") from exc

        try:
            choices = data.get("choices", [])
            if not choices or not isinstance(choices, list):
                raise LLMClientError(f"Unexpected LLM API response format: {data}")

            message = choices.get("message", {})
            if not isinstance(message, dict):
                raise LLMClientError(f"Unexpected LLM API response format: {data}")

            return self._coerce_message_text(message)
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected LLM API response format: {data}") from exc

    def chat(
        self,
        user_message: str,
        system_prompt: str = "You are a concise assistant for the Adult Income capstone project.",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
