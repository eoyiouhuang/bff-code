import os
import json
from pathlib import Path
from google import genai
from dotenv import load_dotenv

# Load the API key from the .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR.parent / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_turn(transcript, conversation_history=None):
    """
    Takes a transcript (string of what the owner just said)
    and returns a dictionary with VAD scores, tone, topics, and sarcasm flag.
    """

    # Build context from recent conversation history if provided
    history_text = ""
    if conversation_history:
        history_text = "\n".join([
            f"{turn['role']}: {turn['content']}"
            for turn in conversation_history[-5:]
        ])
        history_text = f"\nRecent conversation context:\n{history_text}\n"

    # The prompt we send to Gemini
    prompt = f"""You are an emotion analyst for a robot dog companion system.
Analyze the emotional content of this utterance from the dog's owner.
{history_text}
Owner just said: "{transcript}"

Return ONLY a valid JSON object with exactly these fields:
{{
  "valence": <float between -1.0 (very negative) and 1.0 (very positive)>,
  "arousal": <float between 0.0 (very calm) and 1.0 (very excited)>,
  "dominance": <float between 0.0 (overwhelmed) and 1.0 (in control)>,
  "tone": <one word description of tone, e.g. "warm", "flat", "tense", "playful">,
  "topics": <list of 1-3 topic words, e.g. ["work", "stress"]>,
  "sarcasm_flag": <true or false>,
  "confidence": <float between 0.0 and 1.0>
}}

Return only the JSON. No explanation, no markdown, no code blocks.
"""

    # Retry up to 3 times in case Gemini returns empty or malformed response
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            response_text = response.text.strip()

            # Strip markdown code blocks if Gemini wraps the JSON in them
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json or ```) and last line (```)
                response_text = "\n".join(lines[1:-1]).strip()

            # Parse and return
            result = json.loads(response_text)
            return result

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"  [Retry {attempt+1}/3] Gemini returned unexpected format, retrying...")
            if attempt == 2:
                # All retries failed — return a safe neutral fallback
                print("  [Fallback] Using neutral VAD values.")
                return {
                    "valence": 0.0,
                    "arousal": 0.5,
                    "dominance": 0.5,
                    "tone": "neutral",
                    "topics": [],
                    "sarcasm_flag": False,
                    "confidence": 0.0
                }

# ── TEST: run this file directly to try it out ──
if __name__ == "__main__":
    test_phrases = [
        "I had a really great day today!",
        "I'm fine.",
        "Oh great, another Monday.",
        "I'm so tired, I don't even know what I'm doing anymore.",
    ]

    print("Testing turn_analyzer.py\n" + "="*40)
    for phrase in test_phrases:
        print(f"\nInput: \"{phrase}\"")
        result = analyze_turn(phrase)
        print(f"  Valence:    {result['valence']:+.2f}  (-1=negative, +1=positive)")
        print(f"  Arousal:    {result['arousal']:.2f}  (0=calm, 1=excited)")
        print(f"  Dominance:  {result['dominance']:.2f}  (0=overwhelmed, 1=in control)")
        print(f"  Tone:       {result['tone']}")
        print(f"  Topics:     {result['topics']}")
        print(f"  Sarcasm:    {result['sarcasm_flag']}")
        print(f"  Confidence: {result['confidence']:.2f}")