"""
app.py — Flask web backend for AI Chatbot.
Run with: python app.py  →  open http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, session
from src.chatbot import ChatBot
from src.config import Config
from src.prompts import PromptEngine

Config.validate()

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# Store one ChatBot instance per session via history only (stateless-friendly)
SYSTEM_PROMPT = PromptEngine().get_system_prompt()


def get_bot() -> ChatBot:
    """Reconstruct a ChatBot from session history on every request."""
    bot = ChatBot()
    if "history" in session:
        bot.history = session["history"]
    else:
        session["history"] = bot.history
    return bot


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    bot = get_bot()
    try:
        reply = bot.chat(user_message)
        session["history"] = bot.history
        session.modified = True
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    bot = ChatBot()          # fresh bot resets history
    session["history"] = bot.history
    session.modified = True
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(debug=Config.DEBUG)
