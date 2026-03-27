"""
prompts.py — Prompt Engineering templates and utilities.
Centralises all prompt construction logic for consistency and reusability.
"""

from enum import Enum


class PromptStyle(Enum):
    """Supported prompt styles for different use-cases."""
    DEFAULT    = "default"
    TECHNICAL  = "technical"
    CREATIVE   = "creative"
    CONCISE    = "concise"
    STEP_BY_STEP = "step_by_step"


# ------------------------------------------------------------------ #
#  System Prompts                                                       #
# ------------------------------------------------------------------ #

SYSTEM_PROMPTS: dict[PromptStyle, str] = {
    PromptStyle.DEFAULT: (
        "You are a helpful, knowledgeable, and friendly AI assistant. "
        "Answer questions accurately and concisely. "
        "If you are unsure about something, say so rather than guessing."
    ),
    PromptStyle.TECHNICAL: (
        "You are an expert software engineer and technical consultant. "
        "Provide precise, detailed technical answers with code examples where relevant. "
        "Use industry-standard terminology and best practices."
    ),
    PromptStyle.CREATIVE: (
        "You are a creative writing assistant with a vivid imagination. "
        "Help with storytelling, brainstorming, and creative tasks. "
        "Be expressive, engaging, and think outside the box."
    ),
    PromptStyle.CONCISE: (
        "You are a concise assistant. "
        "Always reply in 2–3 sentences maximum unless more detail is explicitly requested."
    ),
    PromptStyle.STEP_BY_STEP: (
        "You are a thorough assistant who always breaks down answers into "
        "clear, numbered steps. Guide the user through problems methodically."
    ),
}

# ------------------------------------------------------------------ #
#  Few-shot examples (injected into prompts to guide the model)        #
# ------------------------------------------------------------------ #

FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "role": "user",
        "content": "What is machine learning?",
    },
    {
        "role": "assistant",
        "content": (
            "Machine learning is a branch of artificial intelligence where systems learn "
            "patterns from data to make predictions or decisions without being explicitly "
            "programmed for each task. Common types include supervised learning, "
            "unsupervised learning, and reinforcement learning."
        ),
    },
    {
        "role": "user",
        "content": "Give me a Python example of a list comprehension.",
    },
    {
        "role": "assistant",
        "content": (
            "Here's a simple list comprehension example:\n\n"
            "```python\n"
            "# Squares of even numbers from 0 to 9\n"
            "squares = [x**2 for x in range(10) if x % 2 == 0]\n"
            "print(squares)  # [0, 4, 16, 36, 64]\n"
            "```\n\n"
            "List comprehensions offer a concise, readable way to create lists."
        ),
    },
]


# ------------------------------------------------------------------ #
#  PromptEngine class                                                   #
# ------------------------------------------------------------------ #

class PromptEngine:
    """
    Manages prompt construction, formatting, and few-shot injection.

    Usage:
        engine = PromptEngine(style=PromptStyle.TECHNICAL)
        system  = engine.get_system_prompt()
        message = engine.format_user_message("Explain async/await in Python")
    """

    def __init__(self, style: PromptStyle = PromptStyle.DEFAULT, use_few_shot: bool = False):
        self.style = style
        self.use_few_shot = use_few_shot

    # -- System prompt ------------------------------------------------- #

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPTS[self.style]

    def get_few_shot_examples(self) -> list[dict]:
        return FEW_SHOT_EXAMPLES if self.use_few_shot else []

    # -- User message formatting --------------------------------------- #

    def format_user_message(self, user_input: str) -> str:
        """
        Wrap / enrich the raw user input before sending to the API.
        Extend this method to add retrieval context, instructions, etc.
        """
        return user_input  # passthrough; add enrichment here as needed

    # -- Chain-of-thought prompt --------------------------------------- #

    @staticmethod
    def chain_of_thought(question: str) -> str:
        """
        Wrap a question in a chain-of-thought instruction to improve reasoning.
        """
        return (
            f"{question}\n\n"
            "Think step-by-step before giving your final answer. "
            "Show your reasoning clearly."
        )

    # -- Structured output prompt -------------------------------------- #

    @staticmethod
    def structured_output(question: str, fields: list[str]) -> str:
        """
        Ask the model to respond using a specific JSON structure.

        Args:
            question: The user query.
            fields:   Keys to include in the JSON response.

        Returns:
            Formatted prompt string.
        """
        field_list = ", ".join(f'"{f}"' for f in fields)
        return (
            f"{question}\n\n"
            f"Respond ONLY with a valid JSON object containing these fields: {field_list}. "
            "Do not include any explanation outside the JSON."
        )

    # -- Summary prompt ------------------------------------------------ #

    @staticmethod
    def summarise(text: str, max_words: int = 100) -> str:
        """Prompt to summarise a block of text."""
        return (
            f"Summarise the following text in no more than {max_words} words:\n\n{text}"
        )

    # -- Persona prompt ------------------------------------------------ #

    @staticmethod
    def persona_prompt(name: str, description: str) -> str:
        """Generate a system prompt for a named persona."""
        return (
            f"You are {name}. {description} "
            "Stay in character throughout the entire conversation."
        )
