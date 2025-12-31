# Testing the LLM translation on only 1 question

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Schema_Extractor import extract_schema, build_prompt


# ---------- CONFIG ----------
MODEL_NAME = "prithivMLmods/Qwen-UMLS-7B-Instruct"
KG_PATH = "/home/abdelmoula/neo4j/import/kg.csv"   # Path to PrimeKG CSV
QUESTION = "Name all gene/proteins that are associated with drug Minoxidil."
# ---------------------------- Solution for '/ issue'
import re
def canon_gp(text: str) -> str:
    text = re.sub(r'genes?\s*/\s*proteins?', 'gene/protein', text, flags=re.I)
    text = re.sub(r'genes?\s+and\s+proteins?', 'gene/protein', text, flags=re.I)
    text = re.sub(r'gene\s*/\s*proteins?', 'gene/protein', text, flags=re.I)
    return text

QUESTION = canon_gp(QUESTION)
# ---------- 1️⃣ Load Model ----------
print(f"🔹 Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ Model loaded successfully.\n")

# ---------- 2️⃣ Build Schema Context ----------
schema_info = extract_schema(KG_PATH)
base_prompt = build_prompt(schema_info)



# ---------- 3️⃣ One-Shot Example ---------- Version 1
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

# ---------- 4️⃣ Build Final One-Shot Prompt ----------
prompt = f"""{base_prompt}

{FEWSHOT_EXAMPLE}
User: {QUESTION}
Cypher:
Please write only ONE Cypher query that answers the User's question.
Do NOT generate more examples or explanations.
End your answer immediately after the Cypher query.
"""

print("🧱 === Prompt Preview ===")
print(prompt[:1200], "...\n")

# ---------- 5️⃣ Define LLM Query Helper ----------
def query_llm(prompt, max_new_tokens=250, temperature=0.3):
    """Runs the LLM on the given prompt and extracts only the Cypher query."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🧹 Remove the echoed prompt
    if result.startswith(prompt.strip()):
        result = result[len(prompt.strip()):].strip()

    # 🧹 Cut off any hallucinated continuations
    for marker in ["Human:", "Example:", "New Task:"]:
        if marker in result:
            result = result.split(marker)[0]

    # 🔍 Keep only Cypher query lines
    lines = [line.strip() for line in result.splitlines() if line.strip().startswith("MATCH")]
    if lines:
        result = lines[-1]  # use last MATCH (the actual answer)
        if not result.strip().endswith(";"):
            result += ";"

    return result.strip()


# ---------- 6️⃣ Run One-Shot Prompting ----------
print("🚀 Running few-shot prompting...\n")
response = query_llm(prompt)

print("\n=== MODEL RESPONSE ===\n")
print(response)
