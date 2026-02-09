import re
import time

# -------------------------------
# Concept Dependency Map
# -------------------------------
DEPENDENCIES = {
    "neural networks": ["gradient descent", "loss function"],
    "backpropagation": ["gradient descent", "chain rule"],
    "deep learning": ["neural networks"]
}

# -------------------------------
# Bridge Lessons (Mini Explainers)
# -------------------------------
BRIDGE_LESSONS = {
    "gradient descent": (
        "Gradient descent is an optimization algorithm used to minimize "
        "the loss function by iteratively adjusting model parameters "
        "in the direction of the steepest descent."
    ),
    "loss function": (
        "A loss function measures how far a model's predictions "
        "are from the actual values. Training aims to minimize this loss."
    ),
    "chain rule": (
        "The chain rule is a calculus principle used to compute derivatives "
        "of composite functions, essential for backpropagation."
    )
}

def extract_concepts(text):
    found = set()
    for concept in DEPENDENCIES.keys():
        if re.search(concept, text, re.IGNORECASE):
            found.add(concept)
    return found

def detect_gaps(found_concepts):
    missing = set()
    for concept in found_concepts:
        prereqs = DEPENDENCIES.get(concept, [])
        for p in prereqs:
            if p not in found_concepts:
                missing.add(p)
    return missing

def main():
    print("📄 Analyzing study content...")
    time.sleep(1)

    with open("sample_input.txt", "r") as f:
        content = f.read()

    print("🧠 Extracting key concepts...")
    time.sleep(1)

    found_concepts = extract_concepts(content)

    print(f"✅ Concepts Found: {', '.join(found_concepts)}")
    time.sleep(1)

    missing = detect_gaps(found_concepts)

    if not missing:
        print("🎉 No knowledge gaps detected!")
        return

    print("\n⚠️ Knowledge Gaps Detected:")
    for m in missing:
        print(f" - {m.title()}")
        time.sleep(0.5)

    print("\n🧩 Injecting Bridge Lessons...\n")
    time.sleep(1)

    print("📘 ENHANCED CONTENT OUTPUT")
    print("-" * 50)
    print(content.strip(), "\n")

    for m in missing:
        lesson = BRIDGE_LESSONS.get(m, "Explanation not available.")
        print(f"🔹 {m.title()} (Bridge Concept):")
        print(lesson)
        print()

    print("-" * 50)
    print("✅ Content successfully enhanced!")

if __name__ == "__main__":
    main()
