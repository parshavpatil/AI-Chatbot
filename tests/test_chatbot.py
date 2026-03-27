"""
tests/test_chatbot.py — Unit tests for the AI Chatbot project.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from src.prompts import PromptEngine, PromptStyle
from src.multi_step_handler import MultiStepHandler


# ------------------------------------------------------------------ #
#  PromptEngine tests                                                   #
# ------------------------------------------------------------------ #

class TestPromptEngine:

    def test_default_system_prompt_is_non_empty(self):
        engine = PromptEngine()
        prompt = engine.get_system_prompt()
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_technical_style_contains_engineer(self):
        engine = PromptEngine(style=PromptStyle.TECHNICAL)
        assert "engineer" in engine.get_system_prompt().lower()

    def test_few_shot_disabled_by_default(self):
        engine = PromptEngine()
        assert engine.get_few_shot_examples() == []

    def test_few_shot_enabled(self):
        engine = PromptEngine(use_few_shot=True)
        examples = engine.get_few_shot_examples()
        assert len(examples) > 0
        assert examples[0]["role"] == "user"

    def test_chain_of_thought_includes_question(self):
        question = "Why is the sky blue?"
        result = PromptEngine.chain_of_thought(question)
        assert question in result
        assert "step" in result.lower()

    def test_structured_output_includes_fields(self):
        result = PromptEngine.structured_output("What is AI?", ["summary", "examples"])
        assert '"summary"' in result
        assert '"examples"' in result

    def test_summarise_includes_word_count(self):
        result = PromptEngine.summarise("Some long text here.", max_words=50)
        assert "50" in result

    def test_persona_prompt(self):
        result = PromptEngine.persona_prompt("Ada", "You are a math tutor.")
        assert "Ada" in result


# ------------------------------------------------------------------ #
#  MultiStepHandler tests                                               #
# ------------------------------------------------------------------ #

class TestMultiStepHandler:

    def _make_handler(self):
        """Return a handler with a mocked bot."""
        bot = MagicMock()
        bot.history = [{"role": "system", "content": "You are helpful."}]
        handler = MultiStepHandler(bot)
        return handler, bot

    def test_is_complex_query_true(self):
        handler, _ = self._make_handler()
        assert handler.is_complex_query("How do I learn Python?")
        assert handler.is_complex_query("Compare Flask and Django")
        assert handler.is_complex_query("Step by step guide to REST APIs")

    def test_is_complex_query_false(self):
        handler, _ = self._make_handler()
        assert not handler.is_complex_query("What is Python?")
        assert not handler.is_complex_query("Hello")

    def test_parse_numbered_list_standard(self):
        text = "1. First item\n2. Second item\n3. Third item"
        result = MultiStepHandler._parse_numbered_list(text)
        assert result == ["First item", "Second item", "Third item"]

    def test_parse_numbered_list_parenthesis(self):
        text = "1) Alpha\n2) Beta"
        result = MultiStepHandler._parse_numbered_list(text)
        assert result == ["Alpha", "Beta"]

    def test_parse_numbered_list_fallback(self):
        """Non-numbered input should return the whole text as a single item."""
        text = "Just a plain sentence."
        result = MultiStepHandler._parse_numbered_list(text)
        assert result == ["Just a plain sentence."]

    def test_resolve_step_calls_api(self):
        handler, bot = self._make_handler()
        bot._call_api.return_value = "An answer."
        result = handler.resolve_step("What is an API?")
        assert result == "An answer."
        bot._call_api.assert_called_once()

    def test_handle_falls_back_on_empty_decomposition(self):
        handler, bot = self._make_handler()
        bot._call_api.return_value = ""    # decompose returns empty
        bot._single_turn.return_value = "Fallback answer."
        result = handler.handle("How do I do X?")
        assert result == "Fallback answer."

    def test_handle_full_pipeline(self):
        handler, bot = self._make_handler()
        # decompose returns two steps, resolve + synthesise return strings
        bot._call_api.side_effect = [
            "1. Sub Q1\n2. Sub Q2",   # decompose
            "Answer to Q1.",           # resolve step 1
            "Answer to Q2.",           # resolve step 2
            "Final synthesised answer.",  # synthesise
        ]
        result = handler.handle("How do I build a REST API?")
        assert result == "Final synthesised answer."
        assert bot.history[-1]["content"] == "Final synthesised answer."


# ------------------------------------------------------------------ #
#  ChatBot integration test (mocked API)                               #
# ------------------------------------------------------------------ #

class TestChatBotIntegration:

    @patch("src.chatbot.OpenAI")
    def test_chat_simple_query(self, mock_openai_cls):
        """Simple query should return the mocked API response."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello there!"))]
        )

        from src.chatbot import ChatBot
        bot = ChatBot()
        response = bot.chat("Hi")
        assert response == "Hello there!"

    @patch("src.chatbot.OpenAI")
    def test_reset_clears_history(self, mock_openai_cls):
        from src.chatbot import ChatBot
        bot = ChatBot()
        bot.history.append({"role": "user", "content": "Test"})
        bot.reset()
        assert len(bot.history) == 1  # only system prompt remains

    @patch("src.chatbot.OpenAI")
    def test_empty_input_returns_message(self, mock_openai_cls):
        from src.chatbot import ChatBot
        bot = ChatBot()
        result = bot.chat("   ")
        assert "valid" in result.lower()
