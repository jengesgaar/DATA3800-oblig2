"""
Chat agent that answers questions about crop yields and weather.

The agent does not know any numbers itself. It calls the functions in
agent_tools.py, which read our own data, and phrases the answer around
what they return. That is the whole point: an agent that makes up a
correlation is worse than no agent, and this one cannot.

Run with:  python agent.py
Needs GEMINI_API_KEY in a .env file in the same folder.

Install:  pip install google-genai python-dotenv
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

import agent_tools

# Change this if the model is not available on your key. Check which models
# you have in Google AI Studio.
MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = """
You answer questions about Norwegian crop yields and weather, for the
DATA3800 project at OsloMet.

Rules you must follow:

- Never state a number that did not come from a tool call. If a tool
  cannot give you the number, say so instead of estimating.
- Call list_options first if you are unsure whether a region or crop
  exists, rather than guessing a name.
- The correlations are modest, around 0.3 at the strongest, and most are
  not significant. Say so when you report them. Weather is one factor
  among many, alongside soil, fertiliser, variety and management.
- A correlation is not a cause. Do not claim that rainfall caused a
  particular harvest to fail.
- Answer in the language the user writes in.
- Keep answers short. Two or three sentences is usually enough.
- Only answer questions about Norwegian crop yields and weather in this
  dataset. For anything else, say that it is outside what this assistant
  covers.
"""

# The functions the agent is allowed to call. The SDK reads the type hints
# and docstrings to work out what each one takes.
TOOLS = [
    agent_tools.list_options,
    agent_tools.get_yield,
    agent_tools.get_crop_correlations,
    agent_tools.compare_crops,
]


def build_client():
    """Reads the api key from .env and returns a Gemini client."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit(
            "No GEMINI_API_KEY found.\n"
            "Create a .env file next to this script containing:\n"
            "    GEMINI_API_KEY=your-key-here"
        )
    return genai.Client(api_key=key)


def show_tool_calls(response, already_shown=0):
    """
    Prints which tools were called since the last question.

    The history accumulates across the whole chat, so we skip the entries
    we have already printed and return the new count.

    response (GenerateContentResponse):
        -what chat.send_message returned
    already_shown (int):
        -how many history entries were printed last time
    """
    history = getattr(response, "automatic_function_calling_history", None) or []
    calls = []
    for content in history[already_shown:]:
        for part in content.parts or []:
            if getattr(part, "function_call", None):
                fc = part.function_call
                calls.append(f"{fc.name}({dict(fc.args)})")
    if calls:
        print("   [tools used: " + "; ".join(calls) + "]")
    return len(history)


def main():
    client = build_client()
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            tools=TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    print("Crop yield assistant. Ask about yields, weather and crops.")
    print("Examples:")
    print("  What was the barley yield in Hedmark in 2018?")
    print("  Which crop is least affected by a long dry spell?")
    print("  Hva påvirker hveteavlingen mest?")
    print("Type 'quit' to exit.\n")

    shown = 0

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        try:
            response = chat.send_message(question)
        except Exception as error:
            if "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error):
                print("   Daily quota used up (20 requests per model on the "
                      "free tier). Try again tomorrow or switch MODEL.\n")
            else:
                print(f"   Something went wrong: {error}\n")
            continue

        print(f"\n{response.text}\n")
        shown = show_tool_calls(response, shown)
        print()


if __name__ == "__main__":
    main()
