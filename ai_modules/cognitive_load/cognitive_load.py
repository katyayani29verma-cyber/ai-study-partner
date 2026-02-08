"""
Cognitive Load & Focus Tracking Module
-------------------------------------
This module analyzes webcam-derived attention metrics
(sent from frontend) and determines the student's
cognitive state in a privacy-first way.

No webcam access here.
No images or video processing.
Only numeric + boolean signals.
"""

class CognitiveLoadTracker:
    def __init__(self):
        # Time counters (in seconds)
        self.focused_time = 0
        self.distracted_time = 0
        self.fatigued_time = 0

    def analyze(self, payload: dict) -> dict:
        """
        Expected payload format:

        {
          "user_id": "student_123",
          "session_id": "session_456",
          "metrics": {
              "face_present": true,
              "looking_away_seconds": 12,
              "blink_rate": 22,
              "head_down": false,
              "delta_time": 5
          }
        }
        """

        metrics = payload.get("metrics", {})

        face_present = metrics.get("face_present", True)
        looking_away = metrics.get("looking_away_seconds", 0)
        blink_rate = metrics.get("blink_rate", 15)
        head_down = metrics.get("head_down", False)
        delta_time = metrics.get("delta_time", 5)

        # -----------------------------
        # COGNITIVE STATE RULES
        # -----------------------------
        if not face_present:
            state = "Distracted"
        elif looking_away > 15:
            state = "Distracted"
        elif head_down:
            state = "Distracted"
        elif blink_rate > 25:
            state = "Fatigued"
        else:
            state = "Focused"

        # -----------------------------
        # TIME ACCUMULATION
        # -----------------------------
        if state == "Focused":
            self.focused_time += delta_time
        elif state == "Distracted":
            self.distracted_time += delta_time
        elif state == "Fatigued":
            self.fatigued_time += delta_time

        total_time = (
            self.focused_time +
            self.distracted_time +
            self.fatigued_time
        )

        focus_score = round(
            self.focused_time / total_time, 2
        ) if total_time > 0 else 0.0

        # -----------------------------
        # INTERVENTION LOGIC
        # -----------------------------
        intervention_type = "NONE"
        intervention_message = ""
        break_recommended = False

        if state == "Distracted":
            intervention_type = "REFOCUS"
            intervention_message = "You seem distracted. Try refocusing on the screen."

        elif state == "Fatigued":
            intervention_type = "MICRO_BREAK"
            intervention_message = "You look tired. Consider taking a short break."
            break_recommended = True

        # -----------------------------
        # FINAL RESPONSE (JSON)
        # -----------------------------
        return {
            "cognitive_state": state,
            "focus_score": focus_score,

            "time_breakdown": {
                "focused_seconds": self.focused_time,
                "distracted_seconds": self.distracted_time,
                "fatigued_seconds": self.fatigued_time
            },

            "intervention": {
                "type": intervention_type,
                "message": intervention_message,
                "break_recommended": break_recommended
            }
        }


# -------------------------------------------------
# TERMINAL TEST (SIMULATED WEBCAM INPUT)
# -------------------------------------------------
if __name__ == "__main__":
    tracker = CognitiveLoadTracker()

    simulated_updates = [
        {
            "metrics": {
                "face_present": True,
                "looking_away_seconds": 0,
                "blink_rate": 18,
                "head_down": False,
                "delta_time": 5
            }
        },
        {
            "metrics": {
                "face_present": True,
                "looking_away_seconds": 20,
                "blink_rate": 18,
                "head_down": False,
                "delta_time": 5
            }
        },
        {
            "metrics": {
                "face_present": True,
                "looking_away_seconds": 0,
                "blink_rate": 30,
                "head_down": False,
                "delta_time": 5
            }
        },
        {
            "metrics": {
                "face_present": True,
                "looking_away_seconds": 0,
                "blink_rate": 18,
                "head_down": False,
                "delta_time": 5
            }
        }
    ]

    for i, payload in enumerate(simulated_updates):
        result = tracker.analyze(payload)
        print(f"\nUpdate {i+1}:")
        print(result)
