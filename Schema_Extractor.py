"""
schema_extractor.py
---------------------------------
Extracts graph schema information from PrimeKG CSV (kg.csv),
prompt for LLM-based Cypher query generation.
"""

import pandas as pd
import json
from pathlib import Path

# ==========  CONFIG  ==========
PRIMEKG_CSV = "kg.csv"  # path to your PrimeKG file
# ===============================

def extract_schema(csv_path=PRIMEKG_CSV):
    """Extracts schema information (relations, entity types, mapping)."""
    print(f"📂 Loading KG CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Show info
    print(f"✅ Loaded {len(df):,} triples")
    unique_entity_types = pd.unique(df[['x_type', 'y_type']].values.ravel())
    unique_relations = df['relation'].unique()

    relation_mapping = (
        df.groupby('relation')['display_relation']
        .apply(lambda x: list(set(x)))
        .to_dict()
    )

    return {
        "entity_types": sorted(unique_entity_types.tolist()),
        "relation_mapping": relation_mapping,
        "num_triples": len(df),
        "unique_relations": sorted(unique_relations.tolist())
    }


def build_prompt(schema_info):
    """Creates a textual schema description without task instructions."""
    entity_types = schema_info["entity_types"]
    mapping = schema_info["relation_mapping"]

    mapping_lines = "\n".join([f"{k} -> {v}" for k, v in mapping.items()])

    prompt = f"""
The following is my Neo4j data model.

Unique entity types:
{entity_types}

Relation → Display Relation mapping:
{mapping_lines}

Example triple:
[Quinidine (drug), transporter, SLCO1B1 (gene/protein)]
"""
    return prompt.strip()
