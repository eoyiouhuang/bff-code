"""
Bonding State V2

Stores the robot's current relationship state.

Responsibilities
----------------
• Store relationship variables
• Store owner's current emotional state
• Store recent interaction history
• Store bond history
• Export context for LLM and behavior modules

This class DOES NOT:
• analyze emotions
• call Gemini
• calculate bonding updates
• save/load files

Those responsibilities belong elsewhere.
"""

from datetime import datetime


class BondingState:

    def __init__(self):

        # =====================================================
        # Relationship Dimensions (Long-Term)
        # =====================================================

        self.trust = 0.50
        self.attachment = 0.50
        self.familiarity = 0.00

        # =====================================================
        # Interaction State (Short-Term)
        # =====================================================

        self.comfort = 0.50
        self.curiosity = 0.50
        self.playfulness = 0.50

        # =====================================================
        # Current Perception of Owner
        # (Mirror, not robot emotions)
        # =====================================================

        self.owner_valence = 0.0
        self.owner_arousal = 0.5
        self.owner_dominance = 0.5
        self.owner_tone = "neutral"

        # =====================================================
        # Conversation Memory
        # =====================================================

        self.recent_topics = []
        self.recent_interactions = []

        # =====================================================
        # Bond History
        #
        # Stores WHY relationship values changed.
        # Used for visualization, debugging, research.
        # =====================================================

        self.bond_history = []

        # =====================================================
        # Session Info
        # =====================================================

        self.turn_count = 0
        self.session_start = datetime.now().isoformat()
        self.last_update = self.session_start

    # =====================================================
    # Update Owner Emotion
    # =====================================================

    def update_owner_emotion(
        self,
        valence,
        arousal,
        dominance,
        tone="neutral"
    ):

        self.owner_valence = max(-1.0, min(1.0, float(valence)))
        self.owner_arousal = self._clamp(arousal)
        self.owner_dominance = self._clamp(dominance)
        self.owner_tone = tone

        self.last_update = datetime.now().isoformat()

    # =====================================================
    # Update Relationship Variables
    # =====================================================

    def update_relationship(
        self,
        trust=None,
        attachment=None,
        familiarity=None,
        comfort=None,
        curiosity=None,
        playfulness=None,
        trigger="unknown"
    ):

        before = {
            "trust": self.trust,
            "attachment": self.attachment,
            "familiarity": self.familiarity,
            "comfort": self.comfort,
            "curiosity": self.curiosity,
            "playfulness": self.playfulness,
        }

        if trust is not None:
            self.trust = self._clamp(trust)

        if attachment is not None:
            self.attachment = self._clamp(attachment)

        if familiarity is not None:
            self.familiarity = self._clamp(familiarity)

        if comfort is not None:
            self.comfort = self._clamp(comfort)

        if curiosity is not None:
            self.curiosity = self._clamp(curiosity)

        if playfulness is not None:
            self.playfulness = self._clamp(playfulness)

        after = {
            "trust": self.trust,
            "attachment": self.attachment,
            "familiarity": self.familiarity,
            "comfort": self.comfort,
            "curiosity": self.curiosity,
            "playfulness": self.playfulness,
        }

        delta = {}

        for key in before:
            delta[key] = round(after[key] - before[key], 4)

        self.bond_history.append({

            "timestamp": datetime.now().isoformat(),

            "trigger": trigger,

            "before": before,

            "after": after,

            "delta": delta

        })

        # keep only most recent 500 updates

        self.bond_history = self.bond_history[-500:]

        self.last_update = datetime.now().isoformat()

    # =====================================================
    # Conversation Memory
    # =====================================================

    def add_topic(self, topic):

        if topic is None:
            return

        if topic not in self.recent_topics:
            self.recent_topics.append(topic)

        self.recent_topics = self.recent_topics[-10:]

    def record_interaction(self, summary):

        self.recent_interactions.append({

            "time": datetime.now().isoformat(),

            "summary": summary

        })

        self.recent_interactions = self.recent_interactions[-20:]

    def increment_turn(self):

        self.turn_count += 1

    # =====================================================
    # Relationship Level
    # =====================================================

    def get_relationship_level(self):

        score = (

            self.trust +

            self.attachment +

            self.familiarity

        ) / 3

        if score < 0.25:
            return "stranger"

        elif score < 0.50:
            return "getting_familiar"

        elif score < 0.75:
            return "close_companion"

        else:
            return "deeply_bonded"

    # =====================================================
    # Export for Behavior System
    # =====================================================

    def get_behavior_context(self):

        return {

            "relationship_level":
                self.get_relationship_level(),

            "approach_owner":
                self.trust > 0.50,

            "stay_close":
                self.attachment > 0.60,

            "play":
                self.playfulness > 0.60
                and self.owner_valence > 0.20,

            "gentle":
                self.owner_valence < -0.30,

            "curious":
                self.curiosity > 0.60,

            "comfort":
                round(self.comfort, 2)

        }

    # =====================================================
    # Export for LLM Prompt
    # =====================================================

    def get_prompt_context(self):

        return {

            "relationship_level":
                self.get_relationship_level(),

            "trust":
                round(self.trust, 2),

            "attachment":
                round(self.attachment, 2),

            "familiarity":
                round(self.familiarity, 2),

            "comfort":
                round(self.comfort, 2),

            "owner_state": {

                "valence":
                    round(self.owner_valence, 2),

                "arousal":
                    round(self.owner_arousal, 2),

                "dominance":
                    round(self.owner_dominance, 2),

                "tone":
                    self.owner_tone

            },

            "recent_topics":
                self.recent_topics[-5:]
        }

    # =====================================================
    # Debug Summary
    # =====================================================

    def get_summary(self):

        return {

            "relationship":

                self.get_relationship_level(),

            "trust":

                round(self.trust, 2),

            "attachment":

                round(self.attachment, 2),

            "familiarity":

                round(self.familiarity, 2),

            "comfort":

                round(self.comfort, 2),

            "curiosity":

                round(self.curiosity, 2),

            "playfulness":

                round(self.playfulness, 2),

            "owner_tone":

                self.owner_tone,

            "turns":

                self.turn_count,

            "bond_updates":

                len(self.bond_history)

        }

    # =====================================================
    # Utility
    # =====================================================

    @staticmethod
    def _clamp(value):

        return max(

            0.0,

            min(1.0, float(value))

        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    state = BondingState()

    print("\nInitial State")
    print(state.get_summary())

    state.update_owner_emotion(

        valence=0.8,

        arousal=0.7,

        dominance=0.8,

        tone="happy"

    )

    state.update_relationship(

        trust=0.55,

        attachment=0.58,

        familiarity=0.03,

        comfort=0.60,

        curiosity=0.52,

        playfulness=0.65,

        trigger="positive_conversation"

    )

    state.add_topic("school")

    state.record_interaction(

        "Owner talked about classes."

    )

    state.increment_turn()

    print("\nUpdated State")

    print(state.get_summary())

    print("\nBehavior Context")

    print(state.get_behavior_context())

    print("\nPrompt Context")

    print(state.get_prompt_context())

    print("\nMost Recent Bond Update")

    print(state.bond_history[-1])