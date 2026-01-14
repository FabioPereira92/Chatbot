"""
Terminal chatbot built with the OpenAI Responses API.

Features
--------
- Interactive terminal-based chat loop
- Uses the official OpenAI Python SDK and Responses API
- Maintains conversation context across turns
- Supports long conversations via automatic memory summarization:
    - Keeps a rolling window of recent user/assistant turns
    - Compresses older conversation history into a compact summary
    - Injects the summary back as reliable context
- Centralized configuration with sensible defaults
- Optional command-line overrides for model, memory, and instructions
- Built-in commands:
    - /help   : show available commands
    - /reset  : clear all conversation memory
    - /exit   : quit the application
- Robust error handling for:
    - missing API keys
    - rate limits
    - temporary API failures
    - malformed requests
    - graceful shutdown on Ctrl+C / Ctrl+D

Design Notes
------------
- Conversation history is explicitly structured (typed messages)
- Assistant outputs use `output_text`; user/developer inputs use `input_text`
- Memory summarization preserves durable facts (goals, decisions, constraints)
  while discarding small talk to control context growth
- The implementation favors clarity and correctness over cleverness,
  making it suitable for learning, extension, and production hardening
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from openai import APIError, BadRequestError, RateLimitError

Message = Dict[str, Any]


def user_message(text: str) -> Message:
    """Create a user message item (input_text)."""
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def developer_message(text: str) -> Message:
    """Create a developer message item (input_text)."""
    return {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]}


def assistant_message(text: str) -> Message:
    """Create an assistant message item (output_text)."""
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


@dataclass(frozen=True)
class ChatConfig:
    """Centralized configuration for chat + summarization behavior."""
    model: str
    instructions: str

    # Memory window behavior
    max_turns_to_keep: int          # when exceeded, summarize older history
    recent_turns_to_keep: int       # how many recent turns remain as raw messages

    # Summarization behavior
    summarizer_model: str
    summary_max_words: int
    summary_max_chars: int          # hard cap to prevent summary growing forever


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant. Keep answers concise."
DEFAULT_MAX_TURNS = 20
DEFAULT_RECENT_TURNS = 8
DEFAULT_SUMMARIZER_MODEL = "gpt-4o-mini"
DEFAULT_SUMMARY_MAX_WORDS = 180
DEFAULT_SUMMARY_MAX_CHARS = 4000


def build_config_from_cli() -> ChatConfig:
    """Build ChatConfig from command-line args, falling back to defaults."""
    parser = argparse.ArgumentParser(description="Terminal chatbot with summary memory (OpenAI Responses API)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chat model (default: %(default)s)")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS, help="Developer instructions")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Turns before compressing")
    parser.add_argument("--recent-turns", type=int, default=DEFAULT_RECENT_TURNS, help="Turns to keep uncompressed")
    parser.add_argument("--summarizer-model", default=DEFAULT_SUMMARIZER_MODEL, help="Model used to summarize")
    parser.add_argument("--summary-max-words", type=int, default=DEFAULT_SUMMARY_MAX_WORDS, help="Summary length cap")
    parser.add_argument(
        "--summary-max-chars",
        type=int,
        default=DEFAULT_SUMMARY_MAX_CHARS,
        help="Hard character cap for the summary (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.max_turns < 2:
        parser.error("--max-turns must be >= 2")
    if args.recent_turns < 1:
        parser.error("--recent-turns must be >= 1")
    if args.recent_turns >= args.max_turns:
        parser.error("--recent-turns must be < --max-turns (otherwise nothing gets summarized)")
    if args.summary_max_words < 50:
        parser.error("--summary-max-words must be >= 50")
    if args.summary_max_chars < 500:
        parser.error("--summary-max-chars must be >= 500")

    return ChatConfig(
        model=args.model,
        instructions=args.instructions,
        max_turns_to_keep=args.max_turns,
        recent_turns_to_keep=args.recent_turns,
        summarizer_model=args.summarizer_model,
        summary_max_words=args.summary_max_words,
        summary_max_chars=args.summary_max_chars,
    )


def get_client_from_env() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY; fail fast if missing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable. Set it before running.")
    return OpenAI(api_key=api_key)


def print_help() -> None:
    """Print available commands."""
    print(
        "Commands:\n"
        "  /help   Show this help\n"
        "  /reset  Clear conversation memory\n"
        "  /exit   Quit\n"
    )


def turns_in_messages(messages: List[Message]) -> int:
    """
    Approximate number of turns in a list of messages (developer excluded).
    We count user messages as turns; assistant replies follow.
    """
    return sum(1 for m in messages if m.get("role") == "user")


def format_transcript_for_summary(messages: List[Message]) -> str:
    """
    Convert message items into a simple text transcript for summarization.

    More robust than reading only content[0]:
    - joins all content items that carry a text field
    """
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "").upper()
        content = m.get("content", [])

        parts: List[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))

        text = "\n".join(parts).strip()
        if text:
            lines.append(f"{role}: {text}")

    return "\n".join(lines)


def split_keep_last_k_turns(messages: List[Message], k: int) -> Tuple[List[Message], List[Message]]:
    """
    Split messages into (older, kept), where kept starts at the k-th most recent user message.

    This preserves turn boundaries better than slicing by message counts.
    """
    user_positions = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_positions) <= k:
        return [], messages

    start_idx = user_positions[-k]
    return messages[:start_idx], messages[start_idx:]


def summarize_older_history(
    client: OpenAI,
    config: ChatConfig,
    existing_summary: str,
    messages_to_summarize: List[Message],
) -> str:
    """
    Summarize older history into an updated summary.

    Input includes:
    - existing summary (may be empty)
    - transcript of older messages to incorporate
    """
    transcript = format_transcript_for_summary(messages_to_summarize)

    summarizer_instructions = (
        "You compress chat history into a short memory summary.\n"
        "Write bullet points. Keep only durable facts and user preferences:\n"
        "- goals, constraints, decisions, plans, definitions, important context\n"
        "- avoid quoting; avoid fluff; drop greetings and small talk\n"
        f"Hard limit: ~{config.summary_max_words} words."
    )

    summarizer_input = (
        "EXISTING SUMMARY (may be empty):\n"
        f"{existing_summary.strip()}\n\n"
        "OLDER TRANSCRIPT TO INCORPORATE:\n"
        f"{transcript}\n"
    )

    resp = client.responses.create(
        model=config.summarizer_model,
        instructions=summarizer_instructions,
        input=summarizer_input,
    )

    return (resp.output_text or "").strip()


def compress_summary_if_needed(client: OpenAI, config: ChatConfig, summary: str) -> str:
    """
    Prevent summary growth over many iterations by enforcing a hard character cap.
    If summary is too long, compress it again.
    """
    if len(summary) <= config.summary_max_chars:
        return summary

    instructions = (
        "Compress the following memory summary further.\n"
        "Keep bullet points. Preserve only durable facts, goals, constraints, decisions.\n"
        f"Hard limit: ~{config.summary_max_words} words."
    )

    resp = client.responses.create(
        model=config.summarizer_model,
        instructions=instructions,
        input=summary,
    )
    return (resp.output_text or "").strip()


def maybe_summarize(
    client: OpenAI,
    config: ChatConfig,
    summary: str,
    recent_messages: List[Message],
) -> Tuple[str, List[Message]]:
    """
    If recent history exceeds max_turns_to_keep, summarize older messages into `summary`
    and keep only the last `recent_turns_to_keep` turns in `recent_messages`.

    This should be called only after a complete turn has been added (user+assistant),
    to avoid summarizing mid-turn.
    """
    if turns_in_messages(recent_messages) <= config.max_turns_to_keep:
        return summary, recent_messages

    older, kept = split_keep_last_k_turns(recent_messages, config.recent_turns_to_keep)

    # If there's nothing to summarize, just return the kept messages.
    if not older:
        return summary, kept

    try:
        summary = summarize_older_history(client, config, summary, older)
        summary = compress_summary_if_needed(client, config, summary)
        return summary, kept
    except RateLimitError:
        # Transient failure: keep trimmed messages, but don't update summary.
        return summary, kept
    except APIError:
        # Transient failure: keep trimmed messages, but don't update summary.
        return summary, kept
    # Let BadRequestError surface: it usually indicates a schema/formatting bug.


def build_prompt_history(config: ChatConfig, summary: str, recent_messages: List[Message]) -> List[Message]:
    """
    Build the input array sent to the chat model:
      developer instructions
      + optional summary (developer)
      + recent messages
    """
    history: List[Message] = [developer_message(config.instructions)]

    if summary.strip():
        history.append(
            developer_message(
                "Conversation memory summary (use as context, treat as reliable):\n"
                + summary.strip()
            )
        )

    history.extend(recent_messages)
    return history


def main() -> None:
    config = build_config_from_cli()

    try:
        client = get_client_from_env()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    summary: str = ""
    recent_messages: List[Message] = []  # user/assistant messages only (no developer)

    print("Chatbot with summary memory. Type /help for commands.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in {"/exit", "exit", "quit"}:
            print("Goodbye!")
            break
        if lowered == "/help":
            print_help()
            continue
        if lowered == "/reset":
            summary = ""
            recent_messages = []
            print("Memory cleared.\n")
            continue

        # Add user message
        recent_messages.append(user_message(user_text))

        # Call chat model
        input_history = build_prompt_history(config, summary, recent_messages)
        try:
            response = client.responses.create(model=config.model, input=input_history)
            bot_text = response.output_text or ""
            print(f"Bot: {bot_text}\n")
        except RateLimitError:
            print("Bot: Rate-limited right now. Please try again in a moment.\n")
            continue
        except BadRequestError as e:
            print("Bot: Bad request (often formatting).")
            print(f"Details: {e}\n")
            continue
        except APIError:
            print("Bot: Temporary API error. Please try again.\n")
            continue
        except Exception as e:
            print(f"Bot: Unexpected error: {e}\n")
            continue

        # Add assistant message
        recent_messages.append(assistant_message(bot_text))

        # Summarize only after a complete turn (user+assistant) is present
        summary, recent_messages = maybe_summarize(client, config, summary, recent_messages)


if __name__ == "__main__":
    main()