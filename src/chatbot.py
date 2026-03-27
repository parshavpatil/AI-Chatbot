"""
chatbot.py — Core chatbot engine using Groq API.
Handles conversation memory, prompt engineering, and multi-step query handling.
"""

import os
from groq import Groq
from typing import Optional
from src.prompts import PromptEngine
from src.multi_step_handler import MultiStepHandler
from src.config import Config


class ChatBot:
    """
    AI Chatbot powered by Groq's LLM API (free tier).
    Supports conversation history, prompt engineering, and multi-step query handling.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.MODEL
        self.max_tokens = Config.MAX_TOKENS
        self.temperature = Config.TEMPERATURE

        self.prompt_engine = PromptEngine()
        self.multi_step_handler = MultiStepHandler(self)

        self.history: list[dict] = []

        system = system_prompt or self.prompt_engine.get_system_prompt()
        self.history.append({"role": "system", "content": system})

    def _call_api(self, messages: list[dict]) -> str:
        """Send messages to Groq and return the assistant reply."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def chat(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "Please enter a valid query."
        if self.multi_step_handler.is_complex_query(user_input):
            return self.multi_step_handler.handle(user_input)
        return self._single_turn(user_input)

    def _single_turn(self, user_input: str) -> str:
        formatted = self.prompt_engine.format_user_message(user_input)
        self.history.append({"role": "user", "content": formatted})
        reply = self._call_api(self.history)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        system = self.history[0]
        self.history = [system]

    def get_history(self) -> list[dict]:
        return self.history[1:]


def run_cli():
    """Interactive command-line chatbot session."""
    print("\n" + "=" * 55)
    print("   AI Chatbot  |  Powered by Groq (Free)")
    print("=" * 55)
    print("Commands:  'reset' - clear history  |  'quit' - exit\n")

    bot = ChatBot()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            bot.reset()
            print("Conversation reset.")
            continue

        response = bot.chat(user_input)
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    run_cli()
