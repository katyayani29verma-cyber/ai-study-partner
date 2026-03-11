import time

KEY_CONCEPTS = {
    "gradient descent": [
        "loss",
        "minimize",
        "direction",
        "step",
        "update"
    ]
}

FLASHCARDS = {
    "loss": "What does gradient descent minimize?\nA: The loss function.",
    "direction": "In which direction does gradient descent move?\nA: The direction of steepest descent.",
    "update": "What gets updated in gradient descent?\nA: Model parameters."
}

def evaluate_answer(answer, required_keywords):
    found = []
    missing = []

    for k in required_keywords:
        if k in answer.lower():
            found.append(k)
        else:
            missing.append(k)

    score = int((len(found) / len(required_keywords)) * 100)
    return score, found, missing

def main():
    print("\n🎙️ Multi-Modal Revision Engine Initialized")
    time.sleep(1)

    topic = "Gradient Descent"
    print(f"📘 Topic: {topic}\n")
    time.sleep(1)

    print("🤔 Question:")
    print("Explain gradient descent in simple terms.\n")

    answer = input("👉 Your Answer:\n> ")

    print("\n🧠 AI Evaluation...")
    time.sleep(1.5)

    score, found, missing = evaluate_answer(
        answer,
        KEY_CONCEPTS["gradient descent"]
    )

    level = "GOOD"
    if score < 40:
        level = "POOR"
    elif score < 70:
        level = "PARTIAL"

    print(f"\n📊 Understanding Level: {level} ({score}%)")

    if missing:
        print("\n❌ Missing Concepts:")
        for m in missing:
            print(f"- {m}")

        print("\n💡 AI Feedback:")
        print("Your explanation captures the intuition but misses key technical details.")

        print("\n🗂 Generated Flashcards:\n")
        for m in missing:
            if m in FLASHCARDS:
                print("📌 Flashcard")
                print(FLASHCARDS[m])
                print()

    else:
        print("\n✅ Excellent explanation! No gaps detected.")

    print("🏁 Revision cycle complete.\n")

if __name__ == "__main__":
    main()
