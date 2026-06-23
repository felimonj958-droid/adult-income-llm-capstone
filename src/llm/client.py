import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class LLMClientError(Exception):
    pass


class LLMClient:
    """
    Minimal HTTP client wrapper around a chat-style LLM API.

    Expects environment variables:
    - LLM_API_BASE_URL: e.g. https://api.tokenfactory.nebius.com
    - NEBIUS_API_KEY: your Nebius API key
    - LLM_MODEL_NAME: e.g. deepseek-ai/DeepSeek-V3.2
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

            if lines and lines.startswith("```"):
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

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 250,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"

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
            if not choices:
                raise LLMClientError(f"Unexpected LLM API response format: {data}")

            message = choices[0].get("message", {})
            content = message.get("content")


            if content is None or not str(content).strip():
                raise LLMClientError(f"LLM API returned an empty message content: {data}")

            return str(content).strip()
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
