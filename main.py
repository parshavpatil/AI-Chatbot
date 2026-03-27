"""
main.py — CLI entry point.
Run: python main.py
"""

from src.config import Config
from src.chatbot import run_cli

if __name__ == "__main__":
    Config.validate()
    run_cli()
