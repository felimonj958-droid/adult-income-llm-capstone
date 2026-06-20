from src.llm.client import LLMClient

def main():
    client = LLMClient()
    response = client.chat(
        user_message="Say: DeepSeek smoke test passed.",
        system_prompt="You are a concise assistant.",
        temperature=0,
        max_tokens=20,
    )
    print(response)


if __name__ == "__main__":
    main()
