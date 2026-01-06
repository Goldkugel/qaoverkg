# split_biohopr_1hop.py
# ------------------------------------------------------------
# Preprocessing Step 2:
# Randomly splits the 1-hop BioHopR dataset into:
#   - 40% development (dev) set
#   - 60% test set
# ------------------------------------------------------------

import json
import random
from pathlib import Path

# ========= CONFIG =========
SRC_PATH = "data/biohopr_1hop_filtered.json"     # input file (already filtered)
DEV_PATH = "data/dev.json"         # output dev file
TEST_PATH = "data/test.json"       # output test file
DEV_RATIO = 0.4                    # 40% dev / 60% test
SEED = 42                          # for reproducibility
# ===========================

# 1️⃣ Ensure output folder exists
Path("data").mkdir(parents=True, exist_ok=True)

# 2️⃣ Load dataset
print(f"📥 Loading data from {SRC_PATH} ...")
with open(SRC_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

n_total = len(data)
print(f"✅ Loaded {n_total} examples")

# 3️⃣ Shuffle dataset for randomness
random.seed(SEED)
random.shuffle(data)

# 4️⃣ Split indices according to ratio
split_idx = int(n_total * DEV_RATIO)
dev_data = data[:split_idx]
test_data = data[split_idx:]

# 5️⃣ Save both sets
with open(DEV_PATH, "w", encoding="utf-8") as f:
    json.dump(dev_data, f, ensure_ascii=False, indent=2)
with open(TEST_PATH, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# 6️⃣ Print summary
print("📊 Split Summary")
print(f"   Development (40%): {len(dev_data)} examples  →  {DEV_PATH}")
print(f"   Test (60%):        {len(test_data)} examples  →  {TEST_PATH}")
print(f"   Total:             {n_total} examples ✅")
