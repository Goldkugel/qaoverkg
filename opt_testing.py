# Benchmarking the whole dataset + evaluate the LLM's performance
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import json
import re
import sys
import torch
import matplotlib
import time

matplotlib.use('Agg')  # Non-GUI backend

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from neo4j import GraphDatabase
from Schema_Extractor import extract_schema, build_prompt

# -----------------------------
# CONFIG
# -----------------------------
CLEAN_DEV_PATH = "clean_dev.json"
CLEAN_TEST_PATH = "clean_test.json"
MODEL_NAME = "prithivMLmods/Qwen-UMLS-7B-Instruct"
BIG_MODEL_NAME = "/homeshare/abdelmoula/models/Qwen2.5-Coder-32B-Instruct"
KG_PATH = "/home/abdelmoula/neo4j/import/kg.csv"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DB = "neo4j"


# -----------------------------
# Canonicalization
# -----------------------------
def canon_gp(text: str) -> str:
    text = re.sub(r'genes?\s*/\s*proteins?', 'gene/protein', text, flags=re.I)
    text = re.sub(r'genes?\s+and\s+proteins?', 'gene/protein', text, flags=re.I)
    text = re.sub(r'gene\s*/\s*proteins?', 'gene/protein', text, flags=re.I)
    return text


# -----------------------------
# FEWSHOT (unchanged)
# -----------------------------
FEWSHOT_EXAMPLE = """Example:
User: Name all gene / proteins that are associated with drug Caplacizumab.
Cypher:
MATCH (dr:drug)
WHERE toLower(dr.name) = toLower('Caplacizumab')
MATCH (dr)-[:drug_protein]->(gp:`gene/protein`)
RETURN DISTINCT coalesce(gp.symbol, gp.name, gp.id) AS gene_protein;
---
Example:
User: Name all diseases that are treated by drug Oxaliplatin.
Cypher:
MATCH (dr:drug)
WHERE toLower(dr.name) = toLower('Oxaliplatin')
MATCH (dr)-[:indication]->(di:disease)
RETURN DISTINCT di.name AS disease;

---
Example:
User: Name all diseases that are related to effect / phenotype Pain.
Cypher:
MATCH (ph:`effect/phenotype`)
WHERE toLower(ph.name) = toLower('Pain')
MATCH (di:disease)-[:disease_phenotype_positive]->(ph)
RETURN DISTINCT di.name AS disease;

---
Example:
User: Name all diseases that are related to gene / protein BCAT2.
Cypher:
MATCH (gp:`gene/protein`)
WHERE toUpper(coalesce(gp.symbol, gp.name, gp.id)) = toUpper('BCAT2')
MATCH (di:disease)-[:disease_protein]->(gp)
RETURN DISTINCT di.name AS disease;

---
New Task:
"""


# -----------------------------
# Cypher extraction
# -----------------------------
def extract_cypher(result: str, prompt: str) -> str:
    if result.startswith(prompt.strip()):
        result = result[len(prompt.strip()):].strip()

    for marker in ["Human:", "Example:", "New Task:"]:
        if marker in result:
            result = result.split(marker)[0]

    matches = re.findall(r"(MATCH[\s\S]+?;)", result, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return result.strip()


# -----------------------------
# Neo4j helpers
# -----------------------------
def check_syntax(tx, query):
    try:
        tx.run("EXPLAIN " + query).consume()
        return True
    except:
        return False


# -----------------------------
# METRICS FOR SEMANTIC EVALUATION
# -----------------------------

def jaccard(pred, gold):
    # Jaccard similarity with an edge-case convention:
    # if both sets are empty, return 1.0 (treat as perfect agreement), since |P∪G|=0 would be undefined.
    if not pred and not gold:
        return 1.0
    intersection = len(pred & gold)
    union = len(pred | gold)
    return intersection / union if union != 0 else 0.0


def recall(pred, gold):
    # Recall with an edge-case convention:
    # if the gold set is empty, return 1.0 (nothing to retrieve), avoiding division by zero.
    if not gold:
        return 1.0
    intersection = len(pred & gold)
    return intersection / len(gold)


def precision(pred, gold):
    # Precision with an edge-case convention:
    # if the predicted set is empty, return 0.0 (no returned answers), avoiding division by zero.
    if len(pred) == 0:
        return 0.0
    return len(pred & gold) / len(pred)


def f1_score(pred, gold):
    # F1 score computed from the precision/recall functions above:
    # if P+R = 0 (typically when precision=recall=0 under the chosen edge-case handling), return 0.0.
    p = precision(pred, gold)
    r = recall(pred, gold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def execute_query(tx, query):
    try:
        recs = list(tx.run(query))
        results = set()
        if recs:
            key = list(recs[0].keys())[0]  # first column
            for r in recs:
                val = r.get(key)
                if isinstance(val, list):
                    results |= {v.lower().strip() for v in val if isinstance(v, str)}
                elif isinstance(val, str):
                    results.add(val.lower().strip())
        return results
    except:
        return None  # runtime error


# -----------------------------
# MAIN
# -----------------------------
def main():
    t0 = time.perf_counter()
    print("📥 Loading clean_test.json...", flush=True)
    with open(CLEAN_TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.\n", flush=True)

    # Load LLM once
    print(f"🔹 Loading model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    print("✅ Model loaded.\n", flush=True)

    # Load schema once
    print("📦 Loading schema...", flush=True)
    schema_info = extract_schema(KG_PATH)
    base_prompt = build_prompt(schema_info)
    print("✅ Schema ready.\n", flush=True)

    # Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Error Metrics
    syntax_crash = 0
    invalid_cypher = 0
    none_answer = 0
    invalid_count = 0
    correct_count = 0
    total_jaccard = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    # Boxplot metrics
    jaccard_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    # Loop through questions
    for idx, item in enumerate(data):
        if idx == 100:
            break
        question = canon_gp(item["question"])
        gold = {a.lower().strip() for a in item["answer"]}

        print(f"\n\n==============================", flush=True)
        print(f"⚡ EXAMPLE {idx + 1}/{len(data)}", flush=True)
        print(f"QUESTION: {question}", flush=True)
        print("==============================", flush=True)

        # Build prompt
        prompt = f"""{base_prompt}

        {FEWSHOT_EXAMPLE}
        User: {question}
        Cypher:

        The Cypher query you output must follow these rules:
        The query must end with a semicolon.
        Do not output anything except the Cypher query.
        Do not output markdown.
        Do not explain.
        Use only the relationship types written in the schema above.
        Do not invent new relationship names.
        Do not add comments or text outside the query.
        Always write gene/protein and effect/phenotype using backticks around the label.
        Example: (gp:`gene/protein`) and (ep:`effect/phenotype`).
        Never output them without backticks.

        Please write only one Cypher query that answers the question.
        End your answer immediately after the semicolon.
        """
        # Run model
        # Parameter in model.generate called do_sample set it to false to get deterministic results
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.3,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        except Exception as e:
            print("🚨 LLM generation error:", e, flush=True)
            continue

        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cypher = extract_cypher(raw, prompt)
        print("\n🔍 ORIGINAL QUESTION:", flush=True)
        print(question, flush=True)

        print("\n💬 GENERATED CYPHER:", flush=True)
        print(cypher, flush=True)

        # Test syntax in Neo4j
        # Validate + execute
        # -----------------------------
        # EVALUATION: RUN & COMPARE
        # -----------------------------
        with driver.session(database=NEO4J_DB) as session:

            # ---- SYNTAX CHECK ----
            try:
                synt_ok = session.execute_read(check_syntax, cypher)
            except Exception as e:
                print(f"❌ Syntax check crashed: {e}")
                jaccard_list.append(0.0)
                recall_list.append(0.0)
                precision_list.append(0.0)
                f1_list.append(0.0)
                invalid_count += 1
                syntax_crash += 1
                continue
            # !!!
            if not synt_ok:
                print("❌ INVALID CYPHER — skipping")
                jaccard_list.append(0.0)
                recall_list.append(0.0)
                precision_list.append(0.0)
                f1_list.append(0.0)
                invalid_count += 1
                invalid_cypher += 1
                continue
            # !!!
            print("✅ Valid syntax — executing...")

            # ---- EXECUTE QUERY ----
            neo_results = session.execute_read(execute_query, cypher)
            # !!!
            if neo_results is None:
                print("❌ Query returned None (runtime failure) — skipping")
                jaccard_list.append(0.0)
                recall_list.append(0.0)
                precision_list.append(0.0)
                f1_list.append(0.0)
                invalid_count += 1
                none_answer += 1
                continue
            # !!!
            # -----------------------------
            # NEW: semantic evaluation
            # -----------------------------
            pred_set = {x.lower().strip() for x in neo_results}
            gold_set = {x.lower().strip() for x in item["answer"]}

            print("\n🔍 GOLD ANSWERS:")
            print(gold_set)

            print("\n🔍 MODEL PREDICTIONS:")
            print(pred_set)

            # ------- Metric 1: Exact Match -------
            exact = (pred_set == gold_set)
            print(f"🔸 Exact Match = {exact}")

            # ------- Metric 2: Jaccard Similarity -------
            jac = jaccard(pred_set, gold_set)
            print(f"🔸 Jaccard similarity = {jac:.3f}")
            jaccard_list.append(jac)
            total_jaccard += jac

            # ------- Metric 3: Recall -------
            rec = recall(pred_set, gold_set)
            print(f"🔸 Recall = {rec:.3f}")
            recall_list.append(rec)
            total_recall += rec
            # ------- Metric 4: Precision -------
            prec = precision(pred_set, gold_set)
            print(f"🔸 Precision = {prec:.3f}")
            precision_list.append(prec)
            total_precision += prec

            # ------- Metric 5: F1 Score -------
            f1_val = f1_score(pred_set, gold_set)
            print(f"🔸 F1 Score = {f1_val:.3f}")
            f1_list.append(f1_val)
            total_f1 += f1_val

            # ------- Decide correctness -------
            accepted = False

            if exact:
                print("🎉 ACCEPTED (EXACT MATCH) ✓")
                accepted = True
            if accepted:
                correct_count += 1
            else:
                print("❌ WRONG ANSWER")
                print(f"Missing: {gold_set - pred_set}")
                print(f"Extra:   {pred_set - gold_set}")

        torch.cuda.empty_cache()
        sys.stdout.flush()
    t1 = time.perf_counter()
    print(f"⏱️ TOTAL runtime: {t1 - t0:.2f}s ({(t1 - t0) / 60:.2f} min)", flush=True)
    N = len(data)
    print(N)
    if N > 0:
        avg_jaccard = total_jaccard / 300
        avg_recall = total_recall / 300
        avg_precision = total_precision / 300
        avg_f1 = total_f1 / 300
    else:
        avg_jaccard = 0.0
        avg_recall = 0.0
        avg_precision = 0.0
        avg_f1 = 0.0

    print("\n==============================")
    print(f"AVERAGE JACCARD: {avg_jaccard:.3f}")
    print(f"AVERAGE RECALL: {avg_recall:.3f}")
    print(f"AVERAGE PRECISION: {avg_precision:.3f}")
    print(f"AVERAGE F1 SCORE:  {avg_f1:.3f}")
    print()
    print("==============================")

    print("\n==============================")
    print("FINAL STATS:")
    print(f"Invalid queries (Syntax Crash): {syntax_crash}")
    print(f"Invalid queries (Invalid Cypher): {invalid_cypher}")
    print(f"Invalid queries (None return Query): {none_answer}")
    print(f"Invalid queries: {invalid_count}")
    print(f"Correct answers: {correct_count}")

    print("==============================\n")

    print("\n==============================")
    print("METRIC LISTS SUMMARY")

    def safe_get(lst, idx):
        try:
            return lst[idx]
        except:
            return None

    print(f"Jaccard list length: {len(jaccard_list)}")
    print(f"  2nd element:       {safe_get(jaccard_list, 1)}")
    print(f"  before last:       {safe_get(jaccard_list, -2)}\n")

    print(f"Recall list length:  {len(recall_list)}")
    print(f"  2nd element:       {safe_get(recall_list, 1)}")
    print(f"  before last:       {safe_get(recall_list, -2)}\n")

    print(f"Precision list length: {len(precision_list)}")
    print(f"  2nd element:         {safe_get(precision_list, 1)}")
    print(f"  before last:         {safe_get(precision_list, -2)}\n")

    print(f"F1 list length:       {len(f1_list)}")
    print(f"  2nd element:        {safe_get(f1_list, 1)}")
    print(f"  before last:        {safe_get(f1_list, -2)}")
    print("==============================\n")

    print(jaccard_list)
    print("==============================\n")
    print(recall_list)
    print("==============================\n")
    print(precision_list)
    print("==============================\n")
    print(f1_list)
    print("==============================\n")
    print(len(jaccard_list), len(recall_list), len(precision_list), len(f1_list))

    # expects these already exist:
    # jaccard_list, recall_list, precision_list, f1_list
    data = [jaccard_list, recall_list, precision_list, f1_list]
    labels = ["Jaccard", "Recall", "Precision", "F1"]

    # Make sure everything is numeric arrays
    data = [np.asarray(x, dtype=float) for x in data]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    # Violin plot (shows distribution even when many values are 1.0)
    vp = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        widths=0.75
    )

    # Optional: make violins semi-transparent + consistent edges
    for body in vp["bodies"]:
        body.set_alpha(0.35)

    for k in ["cmedians", "cbars", "cmins", "cmaxes"]:
        if k in vp:
            vp[k].set_alpha(0.7)

    # Overlay jittered points (so you see the exact mass at 0/1)
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        if arr.size == 0:
            continue
        x = i + rng.uniform(-0.12, 0.12, size=arr.size)
        ax.scatter(x, arr, s=7, alpha=0.25)

        # annotate % of exact 1.0
        p1 = (arr == 1.0).mean() * 100
        ax.text(i, 1.03, f"{p1:.1f}% = 1.0", ha="center", va="bottom", fontsize=9)

    # Axis / labels
    ax.set_title("Evaluation Metrics Distribution (Violin + Points)", fontsize=14)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Scores")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    # Show the bottom line clearly + avoid clipping
    ax.set_ylim(-0.03, 1.08)
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    fig.savefig("metrics_violinplot.png", dpi=300, bbox_inches="tight")
    print("✅ Saved: metrics_violinplot.png")
    plt.show()


if __name__ == "__main__":
    main()
