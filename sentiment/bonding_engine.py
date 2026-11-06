"""
Bonding Engine

Purpose
-------
Converts the owner's emotional state into relationship updates.

Responsibilities
----------------
• Receive VAD output from turn_analyzer.py
• Decide how relationship changes
• Update BondingState
• Generate relationship triggers

This module contains ALL bonding logic.

BondingState only stores state.
"""

from datetime import datetime
from bonding_state import BondingState


class BondingEngine:

    def __init__(self, state: BondingState):

        self.state = state

        # ===================================================
        # Tunable Parameters
        #
        # These are intentionally grouped together so the
        # lab can easily experiment with different values.
        # ===================================================

        # Long-term relationship growth
        self.TRUST_GAIN = 0.05
        self.ATTACHMENT_GAIN = 0.04
        self.FAMILIARITY_GAIN = 0.01

        # Temporary interaction state
        self.COMFORT_GAIN = 0.06
        self.COMFORT_LOSS = 0.08

        self.PLAYFUL_GAIN = 0.05
        self.PLAYFUL_LOSS = 0.05

        self.CURIOSITY_GAIN = 0.03
        self.CURIOSITY_DECAY = 0.01

        # Comfort naturally returns over time
        self.COMFORT_RECOVERY = 0.01

    # ===================================================
    # Main Update Function
    # ===================================================

    def process_turn(self, vad_result):

        """
        Main entry point.

        Parameters
        ----------
        vad_result : dict

        Example

        {
            "valence":0.7,
            "arousal":0.6,
            "dominance":0.8,
            "tone":"warm",
            "topics":["school"],
            "sarcasm_flag":False,
            "confidence":0.92
        }
        """

        # ---------------------------------------------
        # Read analyzer output
        # ---------------------------------------------

        valence = vad_result["valence"]
        arousal = vad_result["arousal"]
        dominance = vad_result["dominance"]

        tone = vad_result.get("tone", "neutral")

        topics = vad_result.get("topics", [])

        confidence = vad_result.get("confidence", 1.0)

        sarcasm = vad_result.get("sarcasm_flag", False)

        # ---------------------------------------------
        # Robot mirrors owner
        # ---------------------------------------------

        self.state.update_owner_emotion(

            valence=valence,

            arousal=arousal,

            dominance=dominance,

            tone=tone

        )

        # ---------------------------------------------
        # Determine relationship update
        # ---------------------------------------------

        update = self._calculate_relationship_update(

            valence,

            arousal,

            dominance,

            confidence,

            sarcasm,

            topics

        )

        # ---------------------------------------------
        # Apply update
        # ---------------------------------------------

        self.state.update_relationship(

            trust=update["trust"],

            attachment=update["attachment"],

            familiarity=update["familiarity"],

            comfort=update["comfort"],

            curiosity=update["curiosity"],

            playfulness=update["playfulness"],

            trigger=update["trigger"]

        )

        # ---------------------------------------------
        # Store topics
        # ---------------------------------------------

        for topic in topics:

            self.state.add_topic(topic)

        self.state.increment_turn()

        return self.state.get_summary()

    # ===================================================
    # Core Relationship Model
    # ===================================================

    def _calculate_relationship_update(

        self,

        valence,

        arousal,

        dominance,

        confidence,

        sarcasm,

        topics

    ):

        """
        Calculates the next relationship state.

        Returns a dictionary of ABSOLUTE values,
        not deltas.
        """

        trust = self.state.trust
        attachment = self.state.attachment
        familiarity = self.state.familiarity

        comfort = self.state.comfort
        curiosity = self.state.curiosity
        playfulness = self.state.playfulness

        # Confidence weighting

        weight = max(0.0, min(1.0, confidence))

        if sarcasm:
            valence = min(valence, -0.2)

        # ------------------------------------------
        # Positive interaction
        # ------------------------------------------

        if valence > 0.2:

            trust += self.TRUST_GAIN * weight

            attachment += self.ATTACHMENT_GAIN * weight

            comfort += self.COMFORT_GAIN * weight

            familiarity += self.FAMILIARITY_GAIN

        # ------------------------------------------
        # Neutral interaction
        # ------------------------------------------

        elif -0.2 <= valence <= 0.2:

            familiarity += self.FAMILIARITY_GAIN

            comfort += self.COMFORT_RECOVERY

        # ------------------------------------------
        # Negative interaction
        # ------------------------------------------

        else:

            # Relationship remains stable

            comfort -= self.COMFORT_LOSS * abs(valence)

            playfulness -= self.PLAYFUL_LOSS

            familiarity += self.FAMILIARITY_GAIN

                # ------------------------------------------
        # Curiosity Model
        # ------------------------------------------

        # New topics make the robot more curious.
        # Repeated conversations slowly reduce curiosity.

        if len(topics) >= 2:
            curiosity += self.CURIOSITY_GAIN

        elif len(topics) == 1:

            if topics[0] not in self.state.recent_topics:
                curiosity += self.CURIOSITY_GAIN
            else:
                curiosity -= self.CURIOSITY_DECAY

        else:
            curiosity -= self.CURIOSITY_DECAY

        # ------------------------------------------
        # Playfulness Model
        # ------------------------------------------

        # Playfulness depends mostly on
        # positive emotion + energy.

        if valence > 0.4 and arousal > 0.5:

            playfulness += self.PLAYFUL_GAIN

        elif valence < -0.3:

            playfulness -= self.PLAYFUL_LOSS

        # ------------------------------------------
        # Dominance Adjustment
        # ------------------------------------------

        # If owner sounds overwhelmed,
        # become gentler.

        if dominance < 0.25:

            comfort -= 0.02

        # ------------------------------------------
        # Clamp Values
        # ------------------------------------------

        trust = self._clamp(trust)
        attachment = self._clamp(attachment)
        familiarity = self._clamp(familiarity)

        comfort = self._clamp(comfort)
        curiosity = self._clamp(curiosity)
        playfulness = self._clamp(playfulness)

        # ------------------------------------------
        # Determine Trigger
        # ------------------------------------------

        trigger = self._determine_trigger(
            valence,
            arousal,
            dominance,
            sarcasm
        )

        return {

            "trust": trust,

            "attachment": attachment,

            "familiarity": familiarity,

            "comfort": comfort,

            "curiosity": curiosity,

            "playfulness": playfulness,

            "trigger": trigger

        }

    # ===================================================
    # Trigger Generator
    # ===================================================

    def _determine_trigger(
        self,
        valence,
        arousal,
        dominance,
        sarcasm
    ):

        """
        Generates a human-readable reason for the update.

        Stored inside BondHistory.
        """

        if sarcasm:
            return "sarcasm_detected"

        if valence > 0.6:

            if arousal > 0.6:
                return "excited_positive"

            return "warm_positive"

        if valence > 0.2:
            return "positive_conversation"

        if valence < -0.6:

            if dominance < 0.3:
                return "owner_distress"

            return "strong_negative"

        if valence < -0.2:
            return "negative_conversation"

        return "neutral_conversation"

    # ===================================================
    # Passive Recovery
    # ===================================================

    def passive_update(self):
        """
        Called every few minutes even if nobody speaks.

        Allows temporary interaction variables
        to slowly recover.
        """

        comfort = self.state.comfort
        playfulness = self.state.playfulness

        comfort += self.COMFORT_RECOVERY

        if playfulness < 0.5:
            playfulness += 0.01

        self.state.update_relationship(

            comfort=self._clamp(comfort),

            playfulness=self._clamp(playfulness),

            trigger="passive_recovery"

        )

    # ===================================================
    # Utility
    # ===================================================

    @staticmethod
    def _clamp(value):

        return max(
            0.0,
            min(1.0, float(value))
        )


# ===========================================================
# TEST
# ===========================================================

if __name__ == "__main__":

    state = BondingState()

    engine = BondingEngine(state)

    examples = [

        {
            "valence": 0.9,
            "arousal": 0.8,
            "dominance": 0.7,
            "tone": "happy",
            "topics": ["school"],
            "sarcasm_flag": False,
            "confidence": 0.95
        },

        {
            "valence": -0.7,
            "arousal": 0.2,
            "dominance": 0.2,
            "tone": "sad",
            "topics": ["work"],
            "sarcasm_flag": False,
            "confidence": 0.90
        },

        {
            "valence": 0.3,
            "arousal": 0.4,
            "dominance": 0.6,
            "tone": "calm",
            "topics": ["robot"],
            "sarcasm_flag": False,
            "confidence": 0.85
        }

    ]

    print("\nBonding Engine Test")
    print("=" * 50)

    for i, vad in enumerate(examples, start=1):

        print(f"\nConversation {i}")

        summary = engine.process_turn(vad)

        print(summary)

    print("\nRelationship History")
    print("=" * 50)

    for entry in state.bond_history:

        print(
            entry["trigger"],
            entry["delta"]
        )