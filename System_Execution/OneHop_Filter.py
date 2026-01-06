#!/usr/bin/env python3
#Preprocessing Step 1: filters out 2-hop questions and leaves only 1-hop questions
"""
biohopr_make_1hop_clean.py
--------------------------
Build a 1-hop-only split from the full BioHopR dump.

Keep if:
  - relation_hop1 is truthy
  - hop1_question is non-empty (trimmed)
  - (optional) query is strictly one hop (exactly one relationship, no '*')

Transform:
  - drop all hop-2 keys (hop2_* and relation_hop2)
  - deduplicate by hop1_question (trimmed), union answers
    * do NOT drop when answers are empty; empty stays empty if all duplicates are empty
"""

# NOTE: Optional validation step.
# Our current pipeline relies on BioHopR hop metadata (relation_hop1, hop1_question, etc.)
# and does not require parsing an explicit query string. However, some dataset exports may
# include a `query` field encoding the intended graph traversal; in that case, enabling
# `--check-cypher` provides an additional consistency check to enforce strict 1-hop queries.


import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# relationship occurrence (e.g., (a)-[:REL]->(b) or (a)<-[:REL]-(b))
REL = re.compile(r'(?:<-)?-\s*\[\s*:[A-Za-z0-9_]+[^\]]*\]\s*-\s*(?:->)?')

def is_strict_one_hop(query: str) -> bool:
    """Exactly one relationship and no var-length '*'."""
    if not query:
        return True  # treat missing as OK (metadata-driven)
    if "*" in query:
        return False
    return len(REL.findall(query)) == 1

def norm_q(s: str) -> str:
    """Trim only (no lowercasing/rewriting)."""
    return (s or "").strip()

def to_hashable(a):
    """Answers should be strings; if not, make them hashable safely."""
    if isinstance(a, (str, int, float, bool)) or a is None:
        return a
    return json.dumps(a, sort_keys=True, ensure_ascii=False)

def from_hashable(h):
    """Inverse of to_hashable for JSON-dumped structures."""
    if isinstance(h, str):
        # try parse JSON-dumped structures; fall back to string
        try:
            v = json.loads(h)
            # if it was truly a JSON string (e.g., "Aspirin"), keep original str
            if isinstance(v, str):
                return v
            return v
        except Exception:
            return h
    return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Full BioHopR JSON file (array of objects)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output 1-hop JSON file")
    ap.add_argument("--check-cypher", action="store_true", help="Enforce query is exactly one hop and no '*'")
    args = ap.parse_args()

    data = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    total = len(data)

    kept_rows = []
    dropped_no_relhop1 = 0
    dropped_empty_q = 0
    dropped_cypher = 0

    for r in data:
        if not r.get("relation_hop1"):
            dropped_no_relhop1 += 1
            continue

        qtext = norm_q(r.get("hop1_question", ""))
        if not qtext:
            dropped_empty_q += 1
            continue

        if args.check_cypher and not is_strict_one_hop(r.get("query", "")):
            dropped_cypher += 1
            continue

        # purge hop2 fields
        for k in list(r.keys()):
            if k == "relation_hop2" or k.startswith("hop2_"):
                r.pop(k, None)

        kept_rows.append(r)

    # dedup by hop1_question (trimmed), union answers (do not drop if empty)
    merged_answers = defaultdict(set)
    representative = {}  # keep first seen as representative metadata

    for r in kept_rows:
        q = norm_q(r["hop1_question"])
        representative.setdefault(q, r)  # first wins
        ans_list = r.get("answer", [])
        for a in (ans_list or []):
            merged_answers[q].add(to_hashable(a))

    out_rows = []
    for q, rep in representative.items():
        # build final answer list (union). If none seen, stays empty list.
        answers = [from_hashable(h) for h in sorted(merged_answers[q], key=lambda x: json.dumps(x, ensure_ascii=False))]
        rep_out = dict(rep)
        rep_out["hop1_question"] = q  # normalized trim
        rep_out["answer"] = answers
        out_rows.append(rep_out)

    Path(args.out_path).write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== BioHopR 1-hop build ===")
    print(f"Input rows           : {total}")
    print(f"Kept rows (pre-dedup): {len(kept_rows)}")
    print(f"Unique questions     : {len(out_rows)}")
    print("--- Dropped (reasons) ---")
    print(f"no relation_hop1     : {dropped_no_relhop1}")
    print(f"empty hop1_question  : {dropped_empty_q}")
    print(f"cypher !one-hop      : {dropped_cypher} (check_cypher={args.check_cypher})")
    print(f"Saved → {args.out_path}")
    print(f"Dedup merges: {len(kept_rows) - len(out_rows)} "
          f"(duplicates collapsed into their question groups)")
    from collections import Counter
    q_counts = Counter(r["hop1_question"].strip() for r in kept_rows)
    dupe_groups = sum(1 for c in q_counts.values() if c > 1)
    dupe_rows = sum(c - 1 for c in q_counts.values() if c > 1)
    print(f"Duplicate groups: {dupe_groups}, duplicate rows merged: {dupe_rows}")
if __name__ == "__main__":
    main()
