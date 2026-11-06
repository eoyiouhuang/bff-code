import os
import json
from datetime import datetime
from google import genai
from dotenv import load_dotenv

# Load API key
load_dotenv(dotenv_path="../.env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Where logs live (matches the existing bff-code log path)
LOGS_DIR = os.path.expanduser("~/bff/logs")
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "../owner_profile.json")


def load_recent_logs(max_sessions=10):
    """
    Reads the most recent session log files from ~/bff/logs/.
    Returns a list of session dictionaries.
    Each session is a list of turns with VAD scores and transcripts.
    """
    if not os.path.exists(LOGS_DIR):
        print(f"[MemoryManager] No logs directory found at {LOGS_DIR}")
        print("[MemoryManager] This is normal if SNAPPER hasn't run yet.")
        return []

    # Get all .jsonl files sorted by modification time (newest first)
    log_files = sorted(
        [f for f in os.listdir(LOGS_DIR) if f.endswith(".jsonl")],
        key=lambda f: os.path.getmtime(os.path.join(LOGS_DIR, f)),
        reverse=True
    )[:max_sessions]

    sessions = []
    for filename in log_files:
        filepath = os.path.join(LOGS_DIR, filename)
        turns = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if turns:
            sessions.append({
                "filename": filename,
                "turns": turns
            })

    print(f"[MemoryManager] Loaded {len(sessions)} past sessions from logs.")
    return sessions


def build_owner_summary(sessions, current_profile):
    """
    Sends recent session data to Gemini and asks it to write
    a concise owner summary. Returns the updated profile dict.
    """
    if not sessions:
        print("[MemoryManager] No sessions to summarize.")
        return current_profile

    # Build a readable summary of recent sessions to send to Gemini
    session_texts = []
    for i, session in enumerate(sessions[:5]):  # Use last 5 sessions max
        turns_text = []
        for turn in session["turns"]:
            # Handle both raw log turns and VAD-enriched turns
            transcript = turn.get("transcript") or turn.get("text") or ""
            vad = turn.get("vad", {})
            if transcript:
                vad_str = ""
                if vad:
                    vad_str = f" [V:{vad.get('valence', '?'):.2f} A:{vad.get('arousal', '?'):.2f}]"
                turns_text.append(f"  - \"{transcript}\"{vad_str}")
        if turns_text:
            session_texts.append(
                f"Session {i+1} ({session['filename']}):\n" +
                "\n".join(turns_text[:10])  # Max 10 turns per session
            )

    sessions_combined = "\n\n".join(session_texts)

    prompt = f"""You are analyzing conversation logs between a person and their robot dog companion called SNAPPER.

Here are the recent conversation sessions:

{sessions_combined}

Based on these conversations, write a concise owner profile. Return ONLY valid JSON with exactly these fields:
{{
  "recurring_topics": <list of up to 5 topic strings the owner talks about most>,
  "dominant_tone": <one word describing their usual emotional tone, e.g. "warm", "anxious", "playful">,
  "energy_pattern": <one phrase describing their typical energy level, e.g. "usually calm", "often tired by evening">,
  "notable_patterns": <one sentence describing any notable emotional or behavioral pattern>,
  "memory_summary": <two sentence narrative summary of the relationship so far>
}}

Return only the JSON. No explanation, no markdown, no code blocks.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        response_text = response.text.strip()

        # Strip markdown if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]).strip()

        gemini_summary = json.loads(response_text)
        print("[MemoryManager] Gemini summary generated successfully.")
        return gemini_summary

    except Exception as e:
        print(f"[MemoryManager] Gemini summarization failed: {e}")
        return {}


def run_on_boot(profile_path=PROFILE_PATH):
    """
    Main function — call this when SNAPPER boots up.
    Reads logs, builds owner summary, updates owner_profile.json.
    Returns the full updated profile.
    """
    print("\n[MemoryManager] Running boot memory consolidation...")

    # Load existing profile
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            profile = json.load(f)
        print(f"[MemoryManager] Existing profile loaded. Affinity: {profile.get('affinity_score', 50)}")
    else:
        profile = {"affinity_score": 50.0}
        print("[MemoryManager] No existing profile. Starting fresh.")

    # Load recent session logs
    sessions = load_recent_logs(max_sessions=10)

    # Build Gemini summary if we have sessions
    if sessions:
        summary = build_owner_summary(sessions, profile)
        profile.update(summary)

    # Always update the boot timestamp
    profile["last_boot"] = datetime.now().isoformat()

    # Save updated profile
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"[MemoryManager] Profile updated and saved to {profile_path}")
    print(f"[MemoryManager] Bonding level: {get_bonding_level(profile.get('affinity_score', 50))}")

    return profile


def consolidate_session(session_vad_history, profile_path=PROFILE_PATH):
    """
    Call this at end of session (when dog enters DREAMING state).
    Writes the session's VAD trajectory into the profile as a memory.
    """
    if not session_vad_history:
        return

    # Load existing profile
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            profile = json.load(f)
    else:
        profile = {}

    # Compute session averages
    avg_v = sum(t["valence"] for t in session_vad_history) / len(session_vad_history)
    avg_a = sum(t["arousal"] for t in session_vad_history) / len(session_vad_history)
    avg_d = sum(t["dominance"] for t in session_vad_history) / len(session_vad_history)

    # Append to session history list
    if "session_history" not in profile:
        profile["session_history"] = []

    profile["session_history"].append({
        "timestamp": datetime.now().isoformat(),
        "turns": len(session_vad_history),
        "avg_valence": round(avg_v, 3),
        "avg_arousal": round(avg_a, 3),
        "avg_dominance": round(avg_d, 3)
    })

    # Keep only last 30 sessions
    profile["session_history"] = profile["session_history"][-30:]

    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"[MemoryManager] Session consolidated. Avg valence: {avg_v:.2f}")


def get_bonding_level(score):
    if score < 25:   return "stranger"
    elif score < 50: return "acquaintance"
    elif score < 75: return "familiar"
    else:            return "bonded"


# ── TEST: run this file directly ──
if __name__ == "__main__":

    print("Testing memory_manager.py")
    print("="*40)

    # Test 1: consolidate a fake session into the profile
    print("\n[Test 1] Consolidating a fake session...")
    fake_session = [
        {"valence": 0.8, "arousal": 0.7, "dominance": 0.6},
        {"valence": -0.2, "arousal": 0.3, "dominance": 0.5},
        {"valence": 0.6, "arousal": 0.5, "dominance": 0.7},
    ]
    consolidate_session(fake_session)
    print("Session consolidated.")

    # Test 2: run boot consolidation
    # (will skip Gemini summary if no logs exist yet — that's fine)
    print("\n[Test 2] Running boot consolidation...")
    profile = run_on_boot()

    print("\n[Test 2] Final profile contents:")
    print(json.dumps(profile, indent=2))