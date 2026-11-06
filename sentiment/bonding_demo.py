"""
BFF Bonding System - Laptop Demo

Purpose
-------
Test the bonding system BEFORE connecting it to the physical robot.

Pipeline
--------
User types message
        ↓
turn_analyzer.py
        ↓
VAD / tone / topics
        ↓
BondingEngine.py
        ↓
BondingState
        ↓
Relationship state
        ↓
Ollama
        ↓
SNAPPER response

This file does NOT use:
- Unitree SDK
- microphone
- speaker
- camera
- robot movement
- robot hardware

Run:
    python bonding_demo.py

Optional:
    python bonding_demo.py --model gemma3n:e2b
    python bonding_demo.py --save

The --save flag saves the final bonding state to owner_profile.json
if BondingState provides save_to_profile().
"""


import os
import sys
import json
import argparse
from pathlib import Path

_proxy_variables = {
    key: os.environ.pop(key)
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    )
    if key in os.environ
}

try:
    import ollama
finally:
    os.environ.update(_proxy_variables)

from dotenv import load_dotenv

from turn_analyzer import analyze_turn
from bonding_state import BondingState
from bonding_engine import BondingEngine


# ============================================================
# Configuration
# ============================================================

# Find the directory containing this script.
BASE_DIR = Path(__file__).resolve().parent

# Try the same .env location used by the existing BFF files.
# Also support a .env file directly inside the bff-code folder.
ENV_PATHS = [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
]

for env_path in ENV_PATHS:
    if env_path.exists():
        load_dotenv(env_path)


DEFAULT_MODEL = os.environ.get(
    "BFF_OLLAMA_MODEL",
    "gemma3n:e2b"
)


DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "BFF_SYSTEM_PROMPT",
    "you are SNAPPER a robot dog. "
    "you do not say woof, whir, tail wag. "
    "answer in 2 sentences or less."
)


# ============================================================
# Ollama
# ============================================================

def query_ollama(
    model_name,
    messages,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    num_predict=100,
    num_ctx=2048,
):
    """
    Send conversation to Ollama and return SNAPPER's response.

    This follows the same basic Ollama configuration used by
    chat-manager.py.
    """

    # Ollama runs locally; do not route its requests through the system proxy.
    client = ollama.Client(
        host="http://127.0.0.1:11434",
        trust_env=False,
    )

    try:
        stream = client.chat(
            model=model_name,
            messages=messages,
            stream=True,
            keep_alive=-1,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        )

        response_parts = []

        for chunk in stream:
            # Ollama Python responses normally expose
            # chunk.message.content
            try:
                content = chunk["message"]["content"]
            except (TypeError, KeyError):
                try:
                    content = chunk.message.content
                except AttributeError:
                    content = ""

            if content:
                response_parts.append(content)

        return "".join(response_parts).strip()

    except Exception as e:
        print(f"\n[Ollama Error] {e}")
        return None


# ============================================================
# Relationship Prompt
# ============================================================

def build_relationship_context(state, vad_result):
    """
    Converts the current bonding state into a small piece of
    context that can be given to Ollama.

    Important:
    This does NOT create a Behavioral Drives Layer.

    It simply tells the LLM about the current relationship
    state so that its response can reflect the relationship.
    """

    # Safely retrieve values from BondingState.
    trust = getattr(state, "trust", 0.5)
    attachment = getattr(state, "attachment", 0.5)
    familiarity = getattr(state, "familiarity", 0.5)
    comfort = getattr(state, "comfort", 0.5)
    curiosity = getattr(state, "curiosity", 0.5)
    playfulness = getattr(state, "playfulness", 0.5)

    # Owner's current emotional state.
    valence = vad_result.get("valence", 0.0)
    arousal = vad_result.get("arousal", 0.5)
    dominance = vad_result.get("dominance", 0.5)
    tone = vad_result.get("tone", "neutral")

    trigger = "unknown"

    # BondingState may store the most recent trigger differently
    # depending on its implementation. Try several safe options.
    if hasattr(state, "last_trigger"):
        trigger = state.last_trigger

    elif hasattr(state, "current_trigger"):
        trigger = state.current_trigger

    elif getattr(state, "bond_history", None):
        try:
            trigger = state.bond_history[-1].get(
                "trigger",
                "unknown"
            )
        except (AttributeError, IndexError):
            pass

    context = f"""
RELATIONSHIP STATE:

Trust: {trust:.2f}
Attachment: {attachment:.2f}
Familiarity: {familiarity:.2f}
Comfort: {comfort:.2f}
Curiosity: {curiosity:.2f}
Playfulness: {playfulness:.2f}

CURRENT OWNER EMOTIONAL STATE:

Valence: {valence:+.2f}
Arousal: {arousal:.2f}
Dominance: {dominance:.2f}
Tone: {tone}

Relationship trigger:
{trigger}

Behavioral instruction:
Respond as SNAPPER while naturally reflecting the current
relationship and the owner's emotional state.

Do not explicitly mention numerical relationship scores.
Do not explain the bonding system.
Do not say that you analyzed the owner's emotions.
Do not invent memories that are not present in the conversation.
Do not become excessively affectionate simply because the
relationship score is high.
"""

    return context.strip()


# ============================================================
# Display Functions
# ============================================================

def print_separator():
    print("\n" + "=" * 65)


def print_vad(vad):
    """
    Display the output from turn_analyzer.py.
    """

    print("\n[Emotion Analysis]")

    print(
        f"  Valence:    {vad.get('valence', 0.0):+.2f}"
    )

    print(
        f"  Arousal:    {vad.get('arousal', 0.5):.2f}"
    )

    print(
        f"  Dominance:  {vad.get('dominance', 0.5):.2f}"
    )

    print(
        f"  Tone:       {vad.get('tone', 'unknown')}"
    )

    topics = vad.get("topics", [])

    if topics:
        print(
            f"  Topics:     {', '.join(topics)}"
        )
    else:
        print("  Topics:     none")

    print(
        f"  Sarcasm:    {vad.get('sarcasm_flag', False)}"
    )

    print(
        f"  Confidence: {vad.get('confidence', 0.0):.2f}"
    )


def print_relationship(state):
    """
    Display the current BondingState.
    """

    print("\n[Relationship State]")

    values = [
        ("Trust", "trust"),
        ("Attachment", "attachment"),
        ("Familiarity", "familiarity"),
        ("Comfort", "comfort"),
        ("Curiosity", "curiosity"),
        ("Playfulness", "playfulness"),
    ]

    for label, attribute in values:

        value = getattr(state, attribute, None)

        if value is not None:
            print(
                f"  {label:<12}: {value:.3f}"
            )

    # Display turn count if available.
    turn_count = getattr(
        state,
        "turn_count",
        getattr(state, "session_turn_count", None)
    )

    if turn_count is not None:
        print(
            f"  {'Turns':<12}: {turn_count}"
        )

    # Display most recent trigger.
    trigger = None

    if hasattr(state, "last_trigger"):
        trigger = state.last_trigger

    elif hasattr(state, "current_trigger"):
        trigger = state.current_trigger

    elif getattr(state, "bond_history", None):

        try:
            trigger = state.bond_history[-1].get("trigger")
        except (AttributeError, IndexError):
            pass

    if trigger:
        print(
            f"  {'Trigger':<12}: {trigger}"
        )


def print_full_summary(state):
    """
    Try to display BondingState's built-in summary if available.
    """

    try:
        summary = state.get_summary()

        print("\n[BondingState Summary]")

        if isinstance(summary, dict):
            print(
                json.dumps(
                    summary,
                    indent=2,
                    ensure_ascii=False
                )
            )
        else:
            print(summary)

    except Exception:
        # Not fatal if get_summary() has a different format.
        pass


# ============================================================
# Main Demo
# ============================================================

def run_demo(model_name, save_state=False):
    """
    Start the interactive laptop bonding demo.
    """

    print_separator()

    print("BFF — SNAPPER Bonding System Demo")

    print_separator()

    print("\nThis is a laptop-only test.")
    print("No robot hardware is connected.")
    print()
    print("Type a sentence and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'state' to view the current relationship state.")
    print("Type 'history' to view bonding history.")
    print()

    print(f"Ollama model: {model_name}")

    print_separator()

    # --------------------------------------------------------
    # Initialize bonding system
    # --------------------------------------------------------

    try:
        state = BondingState()
    except Exception as e:

        print(
            f"\n[ERROR] Could not initialize BondingState:"
        )
        print(e)

        return

    try:
        engine = BondingEngine(state)
    except Exception as e:

        print(
            f"\n[ERROR] Could not initialize BondingEngine:"
        )
        print(e)

        return

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    # This history is used by turn_analyzer.py so Gemini can
    # consider recent conversational context.
    conversation_history = []

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    print("\nInitial relationship state:")

    print_relationship(state)

    print_separator()

    # ========================================================
    # Conversation Loop
    # ========================================================

    while True:

        try:
            user_text = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\n\nExiting demo...")

            break

        # ----------------------------------------------------
        # Empty input
        # ----------------------------------------------------

        if not user_text:
            continue

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if user_text.lower() in {
            "quit",
            "exit",
            "q"
        }:

            print("\nExiting demo...")

            break

        # ----------------------------------------------------
        # State command
        # ----------------------------------------------------

        if user_text.lower() == "state":

            print_relationship(state)
            print_full_summary(state)

            continue

        # ----------------------------------------------------
        # History command
        # ----------------------------------------------------

        if user_text.lower() == "history":

            print("\n[Bond History]")

            history = getattr(
                state,
                "bond_history",
                []
            )

            if not history:

                print("  No bonding history yet.")

            else:

                for i, entry in enumerate(
                    history,
                    start=1
                ):

                    print(
                        f"\n  Turn {i}:"
                    )

                    if isinstance(entry, dict):

                        for key, value in entry.items():

                            print(
                                f"    {key}: {value}"
                            )

                    else:

                        print(
                            f"    {entry}"
                        )

            continue

        # ====================================================
        # STEP 1 — Analyze owner's message
        # ====================================================

        print("\n[1/3] Analyzing owner's emotion...")

        try:

            vad_result = analyze_turn(
                user_text,
                conversation_history=conversation_history[-5:]
            )

        except Exception as e:

            print(
                f"\n[Analyzer Error] {e}"
            )

            print(
                "This turn was not added to the bonding system."
            )

            continue

        # ----------------------------------------------------
        # Show analyzer output
        # ----------------------------------------------------

        print_vad(vad_result)

        # ====================================================
        # STEP 2 — Update bonding system
        # ====================================================

        print("\n[2/3] Updating relationship...")

        try:

            summary = engine.process_turn(
                vad_result
            )

        except Exception as e:

            print(
                f"\n[BondingEngine Error] {e}"
            )

            print(
                "The relationship state was not updated."
            )

            continue

        print_relationship(state)

        # ====================================================
        # STEP 3 — Generate SNAPPER response
        # ====================================================

        print("\n[3/3] SNAPPER is thinking...")

        relationship_context = build_relationship_context(
            state,
            vad_result
        )

        # ----------------------------------------------------
        # Build system message
        # ----------------------------------------------------

        system_message = (
            DEFAULT_SYSTEM_PROMPT
            + "\n\n"
            + relationship_context
        )

        # ----------------------------------------------------
        # Add the current user message to conversation
        # ----------------------------------------------------

        conversation_history.append(
            {
                "role": "user",
                "content": user_text
            }
        )

        # ----------------------------------------------------
        # Messages sent to Ollama
        # ----------------------------------------------------

        ollama_messages = [
            {
                "role": "system",
                "content": system_message
            }
        ]

        # Keep recent conversation only.
        #
        # This prevents the prompt from becoming unnecessarily
        # large while preserving conversational continuity.
        ollama_messages.extend(
            conversation_history[-10:]
        )

        # ----------------------------------------------------
        # Query Ollama
        # ----------------------------------------------------

        response = query_ollama(
            model_name=model_name,
            messages=ollama_messages
        )

        # ----------------------------------------------------
        # Handle Ollama failure
        # ----------------------------------------------------

        if response is None:

            print(
                "\nSNAPPER: [No response generated]"
            )

            # Remove the user message because the turn did
            # not receive a corresponding assistant response.
            if conversation_history:
                conversation_history.pop()

            continue

        # ----------------------------------------------------
        # Add response to conversation history
        # ----------------------------------------------------

        conversation_history.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # ----------------------------------------------------
        # Display response
        # ----------------------------------------------------

        print("\nSNAPPER:")
        print(response)

        # ----------------------------------------------------
        # Optional summary
        # ----------------------------------------------------

        print(
            "\n[Bonding update complete]"
        )

        print_separator()

    # ========================================================
    # End of session
    # ========================================================

    print("\nFinal relationship state:")

    print_relationship(state)

    # --------------------------------------------------------
    # Save only if explicitly requested.
    # --------------------------------------------------------

    if save_state:

        print("\nSaving bonding state...")

        try:

            if hasattr(state, "save_to_profile"):

                state.save_to_profile()

                print(
                    "Bonding state saved successfully."
                )

            else:

                print(
                    "BondingState does not provide "
                    "save_to_profile(); state was not saved."
                )

        except Exception as e:

            print(
                f"Could not save bonding state: {e}"
            )

    else:

        print(
            "\nState was NOT saved."
        )

        print(
            "Use --save if you want to persist the session."
        )

    print("\nBFF bonding demo finished.")
    print_separator()


# ============================================================
# Command Line Interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Laptop-only interactive demo for the "
            "BFF bonding system."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ollama model to use. "
            "Defaults to BFF_OLLAMA_MODEL or "
            "gemma3n:e2b."
        )
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "Save the final bonding state to "
            "owner_profile.json."
        )
    )

    args = parser.parse_args()

    run_demo(
        model_name=args.model,
        save_state=args.save
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()