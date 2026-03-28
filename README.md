# AI Chatbot — Powered by Groq (Free)

A full-stack AI chatbot built with Python, Groq LLM API, and Flask.

Check it live here: [AI Chatbot](https://ai-chatbot-5dgw.onrender.com)

## Project Structure

```
ai-chatbot/
├── app.py                    ← Flask web server
├── main.py                   ← CLI entry point
├── requirements.txt
├── .env
├── .gitignore
│
├── src/
│   ├── chatbot.py            ← Core chatbot engine (shared by CLI + web)
│   ├── prompts.py            ← Prompt engineering templates
│   ├── multi_step_handler.py ← Multi-step query pipeline
│   └── config.py             ← Environment config
│
├── templates/
│   └── index.html            ← Web chat UI (HTML + CSS + JS)
│
└── tests/
    └── test_chatbot.py       ← pytest test suite
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup environment
```bash
.env
# Add your Groq API key from https://console.groq.com
```

### 3a. Run as Web App
```bash
python app.py
# Open http://localhost:5000
```

### 3b. Run as CLI
```bash
python main.py
```



## Features
- Free Groq API (llama-3.3-70b-versatile)
- Conversation memory across turns
- Multi-step query handling (decompose → resolve → synthesise)
- Prompt engineering (styles, few-shot, chain-of-thought)
- Dark-themed web UI with typing indicator
- CLI mode for terminal use
