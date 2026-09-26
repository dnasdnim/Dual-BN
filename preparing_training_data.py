# train_phase1.py  (Phase 0 — data preparation)
"""
Convert prism_bn.json into HuggingFace dataset rows for DUAL-BN Phase 1.

Prompts are copied VERBATIM from the eval script (llm_judge_llama_4pass.py)
so training prompts are byte-identical to evaluation prompts (no train/eval drift).

Each subgraph produces:
  - 1 Pass-1 row  (nodes)
  - 1 Pass-2 row  (states)
  - 1 Pass-3 row  (edges)
  - N Pass-4 rows (one per edge, CPDs)
  - 1 Gen row     (PGM -> text)
"""

import json
from datasets import Dataset
from pathlib import Path
from collections import Counter

# ── Configuration ────────────────────────────────────────────
PRISM_JSON_PATH = "prism_bn.json"
OUTPUT_DIR = "data"
TRAIN_BACKBONES = 40
VAL_BACKBONES = 5
TEST_BACKBONES = 5
SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS — copied VERBATIM from llm_judge_llama_4pass.py (the eval script).
# These MUST stay identical to the eval prompts.
# ══════════════════════════════════════════════════════════════════════════════

PASS1_SYSTEM = """You are an expert at identifying nodes (variables) in Bayesian Networks from natural language.
Extract only the node names. Return ONLY valid JSON, no explanation."""

def pass1_prompt(text):
    return f"""Extract all Bayesian Network nodes (variables) from this text.

Text:
\"\"\"{text}\"\"\"

Return ONLY:
{{"nodes": ["node1", "node2", "node3"]}}

Rules:
- List every variable/node mentioned as a random variable
- Use exact names from the text
- Do not include states, only node names"""

PASS2_SYSTEM = """You are an expert at identifying states (possible values) of Bayesian Network nodes from natural language.
For each node given, extract all possible states. Always include a 'None' state for every node.
Return ONLY valid JSON, no explanation."""

def pass2_prompt(text, nodes):
    return f"""For each Bayesian Network node listed, extract all possible states from this text.

Text:
\"\"\"{text}\"\"\"

Nodes: {json.dumps(nodes)}

Return ONLY:
{{
  "node_states": [
    {{"node": "node name", "states": ["state1", "state2", "None"]}}
  ]
}}

Rules:
- Include every node from the list above
- List all states mentioned in the text for each node
- Always include "None" as a state for every node"""

PASS3_SYSTEM = """You are an expert at identifying directed causal relationships between Bayesian Network nodes.
Given nodes and their states, extract which nodes causally influence which others.
Return ONLY valid JSON, no explanation."""

def pass3_prompt(text, node_states):
    nodes_info = "\n".join(
        f"  - {ns['node']}: {json.dumps(ns['states'])}"
        for ns in node_states
    )
    return f"""Identify directed causal edges (A → B means A causes/influences B) between these nodes.

Text:
\"\"\"{text}\"\"\"

Available nodes and their states:
{nodes_info}

Return ONLY:
{{
  "edges": [
    {{"parent": "cause node", "child": "effect node"}}
  ]
}}

Rules:
- Only use node names from the list above
- Only include edges supported by the text
- An edge parent→child means the parent causally influences the child"""

PASS4_SYSTEM = """You are an expert at extracting Conditional Probability Distributions (CPDs) from natural language.
Given a parent-child node pair with their states, produce the CPD matrix.
Rows = child states, columns = parent states. Each column must sum to 1.0.
Return ONLY valid JSON, no explanation."""

def pass4_prompt(text, parent_node, parent_states, child_node, child_states):
    n_rows = len(child_states)
    n_cols = len(parent_states)
    return f"""Extract the CPD matrix for this causal relationship from the text.

Text:
\"\"\"{text}\"\"\"

Parent "{parent_node}" states (columns): {json.dumps(parent_states)}
Child  "{child_node}"  states (rows):    {json.dumps(child_states)}

Return ONLY:
{{
  "parent": "{parent_node}",
  "child":  "{child_node}",
  "matrix": <{n_rows}x{n_cols} list of lists, each column sums to 1.0>
}}

Rules:
- Matrix is {n_rows} rows x {n_cols} columns
- Row order matches child states exactly: {json.dumps(child_states)}
- Column order matches parent states exactly: {json.dumps(parent_states)}
- Every column must sum to 1.0"""

# Gen direction: no eval counterpart (auxiliary, used only for Phase 2 cycle reward).
GEN_SYSTEM = """You are given a parameterized Bayesian Network. Generate a natural-language description
(approximately 350 words) that mentions every variable, every non-"none" state, and every causal relationship.
Use language strength that reflects the conditional probability magnitudes."""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_messages(system, user, target):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]


def canonicalize_cpd_matrix(cpd_matrix):
    """Replace None entries with 0.0, re-normalize columns to sum to 1.0, round to 4 decimals."""
    if not cpd_matrix:
        return cpd_matrix
    n_rows = len(cpd_matrix)
    n_cols = len(cpd_matrix[0]) if cpd_matrix[0] else 0
    if n_cols == 0:
        return cpd_matrix
    clean = [
        [float(cpd_matrix[r][c]) if (c < len(cpd_matrix[r]) and cpd_matrix[r][c] is not None) else 0.0
         for c in range(n_cols)]
        for r in range(n_rows)
    ]
    for c in range(n_cols):
        col_sum = sum(clean[r][c] for r in range(n_rows))
        if col_sum > 0:
            for r in range(n_rows):
                clean[r][c] /= col_sum
        else:
            for r in range(n_rows):
                clean[r][c] = 1.0 / n_rows
    return [[round(clean[r][c], 4) for c in range(n_cols)] for r in range(n_rows)]


def serialize_pgm(nodes):
    """Serialize a PGM for the Gen direction input."""
    lines = ["Nodes and states:"]
    for node_name in sorted(nodes.keys()):
        lines.append(f'  "{node_name}": {nodes[node_name]["states"]}')
    lines.append("\nRoot node priors:")
    for node_name in sorted(nodes.keys()):
        node = nodes[node_name]
        if node.get("level", 0) == 0 and "prior" in node:
            lines.append(f'  "{node_name}": {node["prior"]}')
    lines.append("\nCausal edges and CPDs:")
    edges = []
    for child in sorted(nodes.keys()):
        for parent in sorted(nodes[child].get("parents", {}).keys()):
            edges.append((parent, child))
    for parent, child in edges:
        cpd_info = nodes[child]["parents"][parent]
        clean_cpd = canonicalize_cpd_matrix(cpd_info["cpd_matrix"])
        lines.append(f'  "{parent}" -> "{child}"')
        lines.append(f'    parent states: {cpd_info["parent_states"]}')
        lines.append(f'    child states:  {nodes[child]["states"]}')
        lines.append(f'    CPD matrix:    {clean_cpd}')
    return "\n".join(lines)


def make_rows(sg_id, sg):
    """Generate Phase 1 training rows. Prompts match the eval script exactly."""
    text = sg["generated_text"]
    nodes = sg["nodes"]
    backbone_id = sg["parent_id"]

    # Edges derived from parent relationships
    gt_edges = [{"parent": p, "child": c}
                for c, info in nodes.items()
                for p in info.get("parents", {}).keys()]

    rows = []

    # ── Pass 1: nodes ── (eval uses gt_names = list(gt_nodes.keys()))
    gt_names = list(nodes.keys())
    user = pass1_prompt(text)
    target = json.dumps({"nodes": gt_names})
    rows.append({
        "messages": build_messages(PASS1_SYSTEM, user, target),
        "row_type": "pass1", "backbone_id": backbone_id, "subgraph_id": sg_id,
    })

    # ── Pass 2: states ── (eval passes nodes as a list of name strings)
    user = pass2_prompt(text, gt_names)
    target = json.dumps({"node_states": [
        {"node": n, "states": nodes[n]["states"]} for n in gt_names
    ]})
    rows.append({
        "messages": build_messages(PASS2_SYSTEM, user, target),
        "row_type": "pass2", "backbone_id": backbone_id, "subgraph_id": sg_id,
    })

    # ── Pass 3: edges ── (eval passes node_states as list of {node, states} dicts)
    node_states = [{"node": n, "states": nodes[n]["states"]} for n in gt_names]
    user = pass3_prompt(text, node_states)
    target = json.dumps({"edges": gt_edges})
    rows.append({
        "messages": build_messages(PASS3_SYSTEM, user, target),
        "row_type": "pass3", "backbone_id": backbone_id, "subgraph_id": sg_id,
    })

    # ── Pass 4: one row per edge ──
    # eval: pass4_prompt(text, parent_node, parent_states, child_node, child_states)
    for child, info in nodes.items():
        child_states = info["states"]
        for parent, parent_info in info.get("parents", {}).items():
            parent_states = parent_info["parent_states"]
            canonical_matrix = canonicalize_cpd_matrix(parent_info["cpd_matrix"])
            user = pass4_prompt(text, parent, parent_states, child, child_states)
            target = json.dumps({
                "parent": parent,
                "child": child,
                "matrix": canonical_matrix,
            })
            rows.append({
                "messages": build_messages(PASS4_SYSTEM, user, target),
                "row_type": "pass4", "backbone_id": backbone_id, "subgraph_id": sg_id,
            })

    # ── Gen: PGM → text ── (no eval counterpart)
    user = (
        f'Given the following parameterized Bayesian Network, generate a natural-language '
        f'description (~350 words) that mentions every variable, every non-"none" state, '
        f'and every causal relationship.\n\n{serialize_pgm(nodes)}'
    )
    rows.append({
        "messages": build_messages(GEN_SYSTEM, user, text),
        "row_type": "gen", "backbone_id": backbone_id, "subgraph_id": sg_id,
    })

    return rows


def backbone_stratified_split(backbones, train_n, val_n, test_n, seed):
    import random
    random.seed(seed)
    bbs = sorted(backbones)
    random.shuffle(bbs)
    return (set(bbs[:train_n]),
            set(bbs[train_n:train_n + val_n]),
            set(bbs[train_n + val_n:train_n + val_n + test_n]))


def sanity_check(bns):
    print("\n=== Sanity Checks ===")
    print(f"Total subgraphs: {len(bns)}")
    backbones = {sg["parent_id"] for sg in bns.values()}
    print(f"Unique backbones (parent_id): {len(backbones)}")
    bb_counts = Counter(sg["parent_id"] for sg in bns.values())
    counts = list(bb_counts.values())
    print(f"Subgraphs per backbone: min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.1f}")
    mismatches = sum(
        1 for sg in bns.values()
        if sum(len(info.get("parents", {})) for info in sg["nodes"].values()) != sg["n_edges"]
    )
    print(f"Edge count mismatches: {mismatches}/{len(bns)}")
    none_cpd = 0
    for sg in bns.values():
        for info in sg["nodes"].values():
            for pinfo in info.get("parents", {}).values():
                if any(x is None for row in pinfo.get("cpd_matrix", []) for x in row):
                    none_cpd += 1
                    break
    print(f"CPD matrices containing None entries: {none_cpd}")
    print()
    return len(backbones) == 50


def main():
    print(f"Loading PRISM-BN data from {PRISM_JSON_PATH}")
    if not Path(PRISM_JSON_PATH).exists():
        print(f"ERROR: File not found at {PRISM_JSON_PATH}"); return

    with open(PRISM_JSON_PATH) as f:
        data = json.load(f)
    if "bayesian_networks" not in data:
        print("ERROR: 'bayesian_networks' key not found"); return

    bns = data["bayesian_networks"]
    if "metadata" in data:
        print(f"Metadata: {data['metadata']}")

    if not sanity_check(bns):
        print("WARNING: Backbone count != 50. Continuing, verify the split.")

    backbones = {sg["parent_id"] for sg in bns.values()}
    train_set, val_set, test_set = backbone_stratified_split(
        backbones, TRAIN_BACKBONES, VAL_BACKBONES, TEST_BACKBONES, SEED
    )
    print(f"\nBackbone split: train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    splits = {"train": [], "val": [], "test": []}
    skipped = 0
    print("\nGenerating training rows...")
    for sg_id, sg in bns.items():
        bb = sg["parent_id"]
        split_name = ("train" if bb in train_set else
                      "val" if bb in val_set else
                      "test" if bb in test_set else None)
        if split_name is None:
            skipped += 1; continue
        try:
            for row in make_rows(sg_id, sg):
                row["split"] = split_name
                splits[split_name].append(row)
        except Exception as e:
            print(f"  Skipping {sg_id}: {type(e).__name__}: {e}")
            skipped += 1
    if skipped:
        print(f"\nTotal skipped: {skipped}")

    print(f"\nSaving datasets to {OUTPUT_DIR}/")
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    for split_name, rows in splits.items():
        if not rows:
            print(f"  {split_name}: NO ROWS"); continue
        Dataset.from_list(rows).save_to_disk(f"{OUTPUT_DIR}/{split_name}")
        n_per_type = Counter(r["row_type"] for r in rows)
        print(f"  {split_name}: {len(rows)} rows total")
        for rt in sorted(n_per_type.keys()):
            print(f"    {rt:8}: {n_per_type[rt]}")


if __name__ == "__main__":
    main()
