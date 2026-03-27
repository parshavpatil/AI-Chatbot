"""
multi_step_handler.py — Multi-step query decomposition and handling.
Breaks complex queries into sub-tasks, resolves each, and synthesises a final answer.
"""

from __future__ import annotations
import re
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chatbot import ChatBot


# Keywords that signal a query needs multi-step handling
COMPLEXITY_SIGNALS: list[str] = [
    "step by step", "step-by-step",
    "how do i", "how to",
    "explain and", "compare",
    "pros and cons", "advantages and disadvantages",
    "first", "then", "finally",
    "multiple", "several",
    "plan", "guide", "tutorial",
    "break down", "breakdown",
]


class MultiStepHandler:
    """
    Detects complex queries and resolves them via a decompose → resolve → synthesise pipeline.

    Pipeline:
        1. Decompose  – ask the LLM to split the query into sub-questions.
        2. Resolve    – answer each sub-question individually.
        3. Synthesise – combine sub-answers into one coherent final response.
    """

    def __init__(self, bot: "ChatBot"):
        self.bot = bot

    # ------------------------------------------------------------------ #
    #  Detection                                                            #
    # ------------------------------------------------------------------ #

    def is_complex_query(self, text: str) -> bool:
        """
        Heuristically decide whether a query warrants multi-step processing.
        Returns True if any complexity signal is found.
        """
        lowered = text.lower()
        return any(signal in lowered for signal in COMPLEXITY_SIGNALS)

    # ------------------------------------------------------------------ #
    #  Decompose                                                            #
    # ------------------------------------------------------------------ #

    def decompose(self, query: str) -> list[str]:
        """
        Ask the LLM to decompose a complex query into a numbered list of sub-questions.
        Returns a list of sub-question strings.
        """
        decompose_prompt = (
            f"Break the following question into a numbered list of smaller, "
            f"self-contained sub-questions that together fully answer the original.\n\n"
            f"Question: {query}\n\n"
            f"Return ONLY the numbered list — no explanations, no extra text."
        )

        raw = self.bot._call_api([
            {"role": "system", "content": "You are a query decomposition assistant."},
            {"role": "user",   "content": decompose_prompt},
        ])

        return self._parse_numbered_list(raw)

    # ------------------------------------------------------------------ #
    #  Resolve                                                              #
    # ------------------------------------------------------------------ #

    def resolve_step(self, sub_question: str) -> str:
        """Answer a single sub-question using the bot's LLM."""
        messages = [
            {"role": "system", "content": self.bot.history[0]["content"]},
            {"role": "user",   "content": sub_question},
        ]
        return self.bot._call_api(messages)

    # ------------------------------------------------------------------ #
    #  Synthesise                                                           #
    # ------------------------------------------------------------------ #

    def synthesise(self, original_query: str, steps: list[dict]) -> str:
        """
        Combine individual step answers into a single cohesive final response.

        Args:
            original_query: The user's original question.
            steps: List of {"question": ..., "answer": ...} dicts.
        """
        steps_text = "\n\n".join(
            f"Step {i+1} — {s['question']}\n{s['answer']}"
            for i, s in enumerate(steps)
        )

        synthesis_prompt = (
            f"Original question: {original_query}\n\n"
            f"Here are the answers to each sub-question:\n\n{steps_text}\n\n"
            f"Now write a single, well-structured final answer that integrates all "
            f"the above information clearly and concisely."
        )

        return self.bot._call_api([
            {"role": "system", "content": self.bot.history[0]["content"]},
            {"role": "user",   "content": synthesis_prompt},
        ])

    # ------------------------------------------------------------------ #
    #  Orchestrator                                                         #
    # ------------------------------------------------------------------ #

    def handle(self, query: str) -> str:
        """
        Full multi-step pipeline:
            decompose → resolve each step → synthesise.
        Stores the final answer in conversation history.
        """
        print("\n[Multi-Step] Decomposing query…")
        sub_questions = self.decompose(query)

        if not sub_questions:
            # Fallback: treat as a normal single-turn query
            return self.bot._single_turn(query)

        steps: list[dict] = []
        for i, sq in enumerate(sub_questions, 1):
            print(f"[Multi-Step] Resolving step {i}/{len(sub_questions)}: {sq}")
            answer = self.resolve_step(sq)
            steps.append({"question": sq, "answer": answer})

        print("[Multi-Step] Synthesising final answer…\n")
        final_answer = self.synthesise(query, steps)

        # Persist to conversation history
        self.bot.history.append({"role": "user",      "content": query})
        self.bot.history.append({"role": "assistant",  "content": final_answer})

        return final_answer

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_numbered_list(text: str) -> list[str]:
        """
        Extract items from a numbered list like:
            1. Item one
            2. Item two
        """
        lines = text.strip().splitlines()
        items: list[str] = []
        for line in lines:
            # Match  "1. text", "1) text", "1 - text"
            match = re.match(r"^\s*\d+[\.\)\-]\s*(.+)", line)
            if match:
                items.append(match.group(1).strip())
        return items if items else [text.strip()]
