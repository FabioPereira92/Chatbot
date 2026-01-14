# Terminal Chatbot with Long-Term Memory (OpenAI Responses API) #

A simple, robust terminal chatbot written in Python using the OpenAI Responses API.
It supports long conversations by automatically summarizing older history into a compact memory.

## Features ##
- Interactive terminal chat
- Uses the official OpenAI Python SDK
- Structured conversation history (typed messages)
- Long-term memory via summarization:
  - Keeps recent turns verbatim
  - Compresses older history into a short summary
  - Injects the summary back as reliable context
- Centralized configuration with sensible defaults
- Optional command-line overrides
- Built-in commands:
  - /help – show commands
  - /reset – clear memory
  - /exit – quit
- Robust error handling and graceful shutdown

## How Memory Works ##
The chatbot maintains context using two layers:
1. Recent window – last N user/assistant turns kept verbatim
2. Summary memory - older history summarized into bullet points
   - Keeps goals, decisions, and constraints
   - Drops greetings and small talk
   - Hard-capped to prevent unbounded growth
  
## Requirements ##
- Python 3.9+
- OpenAI API key

## Installation ##
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/your-username/terminal-chatbot.git
cd terminal-chatbot

python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .\.venv\Scripts\activate     # Windows

pip install openai

```
## Set your API key: ##
```bash
export OPENAI_API_KEY="your_key_here"   # macOS / Linux
setx OPENAI_API_KEY "your_key_here"     # Windows
```

## Usage ##
Run with default settings:
python chatbot.py

## Available commands inside the chat ##
```bash
/help    Show help
/reset   Clear memory
/exit    Quit
```

## Configuration(CLI) ##
python chatbot.py \
  --model gpt-4o-mini \
  --max-turns 20 \
  --recent-turns 8 \
  --summary-max-words 180

## Common Options ##
Option	             Description
- --model	             Chat model
- --instructions	     System / developer instructions
- --max-turns	         Turns before summarization
- --recent-turns	     Turns kept verbatim
- --summary-max-words	 Target summary length
- --summary-max-chars	 Hard summary size cap

## Design Goals ##
- Explicit, typed message structure
- Safe error handling
- Readable and maintainable code
- Easy to extend with streaming, tools, or persistence

## License ##
MIT
