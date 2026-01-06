import json

INPUT_PATH = "data/test.json"
OUTPUT_PATH = "clean_test.json"
#Preprocessing Step 3: Removes unnecessary fields from dev/test Dataset
def build_clean_dev(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clean_entries = []

    for entry in data:
        q = entry.get("hop1_question_multi")
        ans = entry.get("answer")

        # Skip entries without required fields
        if q is None or ans is None:
            continue

        clean_entries.append({
            "question": q,
            "answer": ans
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_entries, f, indent=2, ensure_ascii=False)

    print(f"✅ Created {output_path} with {len(clean_entries)} entries.")


if __name__ == "__main__":
    build_clean_dev(INPUT_PATH, OUTPUT_PATH)
