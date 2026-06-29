import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class LLMClientError(Exception):
    """Raised when the LLM client cannot complete a request."""


class LLMClient:
    DEFAULT_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout_seconds: int | float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_API_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY") or os.getenv("LLM_API_KEY", "")
        self.default_model = default_model or os.getenv("LLM_MODEL_NAME", "")
        self.timeout_seconds = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS

        if not self.base_url:
            raise LLMClientError("LLM_API_BASE_URL is not set.")
        if not self.api_key:
            raise LLMClientError("NEBIUS_API_KEY or LLM_API_KEY is not set.")
        if not self.default_model:
            raise LLMClientError("LLM_MODEL_NAME is not set.")

        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned

        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

        return cleaned

    def parse_json(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_code_fences(text)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"LLM returned invalid JSON: {text}") from exc

        if not isinstance(parsed, dict):
            raise LLMClientError(f"LLM returned JSON that is not an object: {parsed}")

        return parsed

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 250,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMClientError(f"Request to LLM API failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError(f"LLM API did not return valid JSON: {response.text}") from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMClientError(f"Unexpected LLM API response format: {data}")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMClientError(f"Unexpected LLM API response format: {data}")

        if first_choice.get("finish_reason") == "length":
            raise LLMClientError("LLM response was truncated before completion.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LLMClientError(f"Unexpected LLM API response format: {data}")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError(f"LLM API returned empty content: {message}")

        return content.strip()

    def chat(
        self,
        user_message: str,
        system_prompt: str = "You are a concise assistant for the Adult Income capstone project.",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        return self.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_json(
        self,
        user_message: str,
        system_prompt: str = (
            "You are a data extraction engine. "
            "Output your entire response as one valid JSON object only. "
            "Do not include markdown, backticks, commentary, or extra text."
        ),
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        text = self.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return self.parse_json(text)