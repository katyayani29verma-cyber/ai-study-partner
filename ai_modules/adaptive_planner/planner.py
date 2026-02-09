"""
Adaptive Study Planner
----------------------
Adapts study sessions based on:
1. Cognitive load (focus, fatigue, distraction)
2. Topic priority (weak areas, revision needs)

This module decides HOW to study, not WHAT the webcam sees.
"""

class AdaptiveStudyPlanner:
    def __init__(self):
        pass

    def generate_plan(
        self,
        cognitive_data: dict,
        topic_priorities: list,
        base_session_minutes: int = 45
    ) -> dict:
        """
        cognitive_data example:
        {
          "cognitive_state": "Fatigued",
          "focus_score": 0.62,
          "time_breakdown": {
              "focused_seconds": 420,
              "distracted_seconds": 180,
              "fatigued_seconds": 60
          },
          "intervention": {
              "type": "MICRO_BREAK",
              "break_recommended": true
          }
        }

        topic_priorities example:
        [
          {"topic": "Probability", "priority": 0.82},
          {"topic": "Calculus", "priority": 0.67},
          {"topic": "Algebra", "priority": 0.30}
        ]
        """

        cognitive_state = cognitive_data.get("cognitive_state", "Focused")
        focus_score = cognitive_data.get("focus_score", 1.0)
        break_needed = cognitive_data.get("intervention", {}).get("break_recommended", False)

        # -----------------------------
        # SESSION LENGTH ADAPTATION
        # -----------------------------
        session_length = base_session_minutes

        if cognitive_state == "Fatigued":
            session_length = 25  # shorter Pomodoro-style session
        elif cognitive_state == "Distracted":
            session_length = 30
        elif focus_score < 0.5:
            session_length = 20

        # -----------------------------
        # STUDY MODE DECISION
        # -----------------------------
        if cognitive_state == "Fatigued":
            study_mode = "REVIEW"
        elif cognitive_state == "Distracted":
            study_mode = "ACTIVE_RECALL"
        else:
            study_mode = "LEARNING"

        # -----------------------------
        # TOPIC SELECTION
        # -----------------------------
        # Sort topics by priority (high → low)
        sorted_topics = sorted(
            topic_priorities,
            key=lambda x: x["priority"],
            reverse=True
        )

        # Take top 2 topics for the session
        selected_topics = sorted_topics[:2]

        # -----------------------------
        # FINAL STUDY PLAN
        # -----------------------------
        return {
            "session_length_minutes": session_length,
            "study_mode": study_mode,
            "break_before_start": break_needed,

            "topics": [
                {
                    "topic": t["topic"],
                    "recommended_minutes": round(session_length / len(selected_topics))
                }
                for t in selected_topics
            ]
        }


# -------------------------------------------------
# TERMINAL TEST (SIMULATION)
# -------------------------------------------------
if __name__ == "__main__":
    planner = AdaptiveStudyPlanner()

    cognitive_input = {
        "cognitive_state": "Fatigued",
        "focus_score": 0.58,
        "intervention": {
            "break_recommended": True
        }
    }

    topic_priorities = [
        {"topic": "Probability", "priority": 0.82},
        {"topic": "Calculus", "priority": 0.67},
        {"topic": "Algebra", "priority": 0.30}
    ]

    plan = planner.generate_plan(
        cognitive_data=cognitive_input,
        topic_priorities=topic_priorities
    )

    print("\n--- Adaptive Study Plan ---")
    print(plan)
