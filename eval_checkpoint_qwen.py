"""
Run direct full-graph inference or held-out evaluation for a Qwen LoRA checkpoint.

Pass 1: Extract nodes        → Node F1, % nodes matched, extra nodes
Pass 2: Extract states       → State F1, % states matched
Pass 3: Extract edges        → % correct, % spurious connections
Pass 4: Extract CPD matrices → KL divergence

Extraction model : DeepSeek-R1-Distill-Qwen-14B + requested LoRA checkpoint
Judge model      : Llama 3.3 70B (via HuggingFace InferenceClient / Groq)

Direct inference:
  CUDA_VISIBLE_DEVICES=0 python eval_checkpoint_qwen.py \
      --checkpoint outputs/phase2_grpo_complexity/checkpoint-1000 \
      --input-file example.txt --prediction-output prediction.json \
      --load-in-4bit

The direct prediction file uses the same ``metadata`` +
``bayesian_networks`` structure as prism_bn.json.

Held-out test evaluation:
  export HF_TOKEN=hf_xxx
  CUDA_VISIBLE_DEVICES=0 python eval_checkpoint_qwen.py \
      --checkpoint outputs/phase2_grpo_complexity/checkpoint-1000 \
      --limit 50 --output-json phase2_eval.json --output-csv phase2_eval.csv \
      --pgm-output phase2_eval_pgms.json \
      --load-in-4bit
"""

import os, re, json, time, math, argparse, csv
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_from_disk
from huggingface_hub import InferenceClient

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
SCRIPT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_FILE = SCRIPT_DIR / "prism_bn.json"
TEST_DATA_PATH = SCRIPT_DIR / "data/test"

NODE_THRESHOLD = 0.5
STATE_THRESHOLD = 0.5
LAPLACE_EPS = 1e-6
MAX_RETRIES = 3
DELAY = 0.0

# ── Load fine-tuned model (global) ────────────────────────────────────────────

def load_checkpoint_model(checkpoint_path, load_in_4bit=False):
    """Load base model + LoRA adapter from checkpoint."""
    print(f"Loading base model {BASE_MODEL} ...")
    checkpoint_path = Path(checkpoint_path)
    tokenizer_source = (
        checkpoint_path
        if (checkpoint_path / "tokenizer_config.json").exists()
        else BASE_MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)

    print(f"Loading LoRA adapter from {checkpoint_path} ...")
    model = PeftModel.from_pretrained(
        model, str(checkpoint_path), is_trainable=False
    )
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    print("Model loaded and ready for inference.")
    return tokenizer, model

_qwen_tok = None
_qwen_model = None
_checkpoint_label = ""
_print_raw_output = False

# ── Full-graph inference prompt (identical to Phase 2 GRPO) ──────────────────

FULL_GRAPH_SYSTEM = """You are an expert at extracting Bayesian Networks from natural language.
Given a text, extract the full Bayesian network as JSON: nodes (with states), directed
edges, and for each edge a CPD matrix (rows = child states, columns = parent states,
each column sums to 1.0). Return ONLY valid JSON, no explanation."""


def full_graph_prompt(text):
    return f"""Extract the full Bayesian Network from this text.

Text:
\"\"\"{text}\"\"\"

Return ONLY this JSON structure:
{{
  "nodes": [{{"node": "name", "states": ["s1", "s2", "none"]}}],
  "edges": [{{"parent": "name", "child": "name"}}], 
  "cpds":  [{{"parent": "name", "child": "name",
              "matrix": [[...]]}}]
}}

Rules:
- Every node has a "none" state
- Edges use only node names from the nodes list
- Each CPD matrix: rows = child states, columns = parent states, each column sums to 1.0"""

# ── Pass 1 prompts ────────────────────────────────────────────────────────────

PASS1_SYSTEM = """Output ONLY valid JSON. No text before or after. No explanation."""

def pass1_prompt(text):
    return f"""Extract all Bayesian Network nodes (variables) from this text.

Text:
\"\"\"{text}\"\"\"

Format: {{"nodes": ["node1", "node2", "node3"]}}

Output ONLY valid JSON, no explanation."""

# ── Pass 2 prompts ────────────────────────────────────────────────────────────

PASS2_SYSTEM = """Output ONLY valid JSON. No text before or after. No explanation."""

def pass2_prompt(text, nodes):
    return f"""Extract all possible states for each Bayesian Network node from this text.

Text:
\"\"\"{text}\"\"\"

Nodes: {json.dumps(nodes)}

Format: {{"node_states": [{{"node": "node name", "states": ["state1", "state2", "None"]}}]}}

Output ONLY valid JSON, no explanation."""

# ── Pass 3 prompts ────────────────────────────────────────────────────────────

PASS3_SYSTEM = """Output ONLY valid JSON. No text before or after. No explanation."""

def pass3_prompt(text, node_states):
    nodes_info = "\n".join(
        f"  - {ns['node']}: {json.dumps(ns['states'])}"
        for ns in node_states
    )
    return f"""Identify directed causal edges between these nodes.

Text:
\"\"\"{text}\"\"\"

Available nodes and their states:
{nodes_info}

Format: {{"edges": [{{"parent": "cause node", "child": "effect node"}}]}}

Output ONLY valid JSON, no explanation."""

# ── Pass 4 prompts ────────────────────────────────────────────────────────────

PASS4_SYSTEM = """Output ONLY valid JSON. No text before or after. No explanation."""

def pass4_prompt(text, parent_node, parent_states, child_node, child_states):
    n_rows = len(child_states)
    n_cols = len(parent_states)
    return f"""Extract the CPD matrix for this causal relationship from the text.

Text:
\"\"\"{text}\"\"\"

Parent "{parent_node}" states (columns): {json.dumps(parent_states)}
Child  "{child_node}"  states (rows):    {json.dumps(child_states)}

Format: {{"parent": "{parent_node}", "child": "{child_node}", "matrix": [[...]]}}

Output ONLY valid JSON, no explanation."""

# ── Judge prompts (binary scoring) ───────────────────────────────────────────

NODE_JUDGE_SYSTEM = """Output ONLY valid JSON. No text before or after. No explanation."""

def node_judge_prompt(gt_names, pred_names):
    return f"""Match each predicted node name to a ground truth node name.

Ground truth: {json.dumps(gt_names)}
Predicted:    {json.dumps(pred_names)}

Format: {{"matches": [{{"predicted": "pred", "ground_truth": "gt", "score": 1}}], "unmatched_predicted": [], "unmatched_ground_truth": []}}

Output ONLY valid JSON, no explanation."""

STATE_JUDGE_SYSTEM = """Output ONLY valid JSON. No text before or after. No explanation."""

def state_judge_prompt(gt_node, pred_node, gt_states, pred_states):
    return f"""Match predicted states to ground truth states for node "{gt_node}".

Ground truth states: {json.dumps(gt_states)}
Predicted states:    {json.dumps(pred_states)}

Format: {{"state_matches": [{{"predicted": "pred", "ground_truth": "gt", "score": 1}}], "unmatched_predicted": [], "unmatched_ground_truth": []}}

Output ONLY valid JSON, no explanation."""

# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json_robust(raw):
    raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw.strip())
    raw = re.sub(r'\n?```$', '', raw).strip()
    start = raw.find('{')
    if start == -1:
        raise ValueError("No JSON found")

    # Try to find the first valid JSON by matching braces
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == '{':
            depth += 1
        elif raw[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i+1])
                except json.JSONDecodeError:
                    pass

    raise ValueError("No valid JSON found")


def graph_levels(nodes, edges):
    """Compute PRISM-style node levels from a generated directed graph."""
    names = [node["node"] for node in nodes if node.get("node")]
    indegree = {name: 0 for name in names}
    children = {name: [] for name in names}
    for edge in edges:
        parent, child = edge.get("parent"), edge.get("child")
        if parent in children and child in indegree and parent != child:
            children[parent].append(child)
            indegree[child] += 1

    queue = [name for name in names if indegree[name] == 0]
    levels = {name: 0 for name in queue}
    cursor = 0
    while cursor < len(queue):
        parent = queue[cursor]
        cursor += 1
        for child in children[parent]:
            levels[child] = max(levels.get(child, 0), levels[parent] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    # Preserve malformed cyclic predictions instead of dropping their nodes.
    fallback = max(levels.values(), default=-1) + 1
    for name in names:
        levels.setdefault(name, fallback)
    return levels


def prediction_to_dataset_nodes(prediction):
    """Convert nodes/edges/cpds output to prism_bn.json's node-centric schema."""
    raw_nodes = prediction.get("nodes", [])
    names = []
    states = {}
    for node in raw_nodes if isinstance(raw_nodes, list) else []:
        if not isinstance(node, dict):
            continue
        name = str(node.get("node", "")).strip()
        if not name or name in states:
            continue
        node_states = node.get("states", [])
        states[name] = node_states if isinstance(node_states, list) else []
        names.append(name)

    raw_edges = prediction.get("edges", [])
    edges = []
    seen_edges = set()
    for edge in raw_edges if isinstance(raw_edges, list) else []:
        if not isinstance(edge, dict):
            continue
        parent = str(edge.get("parent", "")).strip()
        child = str(edge.get("child", "")).strip()
        key = (parent, child)
        if parent in states and child in states and parent != child and key not in seen_edges:
            edges.append({"parent": parent, "child": child})
            seen_edges.add(key)

    raw_cpds = prediction.get("cpds", [])
    cpds = {}
    for cpd in raw_cpds if isinstance(raw_cpds, list) else []:
        if not isinstance(cpd, dict):
            continue
        key = (str(cpd.get("parent", "")).strip(), str(cpd.get("child", "")).strip())
        matrix = cpd.get("matrix", [])
        if key in seen_edges and isinstance(matrix, list):
            cpds[key] = matrix

    levels = graph_levels(
        [{"node": name, "states": states[name]} for name in names], edges
    )
    parents_by_child = {name: [] for name in names}
    for edge in edges:
        parents_by_child[edge["child"]].append(edge["parent"])

    dataset_nodes = {}
    for name in names:
        parents = {
            parent: {
                "parent_states": states[parent],
                "cpd_matrix": cpds.get((parent, name), []),
            }
            for parent in parents_by_child[name]
        }
        dataset_nodes[name] = {
            "states": states[name],
            "level": levels[name],
            "parents": parents,
        }
        if not parents:
            # The extraction prompts do not ask for root priors.
            dataset_nodes[name]["prior"] = []
    return dataset_nodes


def make_dataset_record(record_id, prediction, text, source=None):
    source = source or {}
    return {
        "id": record_id,
        "parent_id": source.get("id"),
        "source_id": source.get("source_id"),
        "title": source.get("title", "Generated Bayesian network"),
        "domain": source.get("domain", "custom"),
        "actual_text": source.get("actual_text", ""),
        "generated_text": text,
        "nodes": prediction_to_dataset_nodes(prediction),
        "generation_metadata": {
            "checkpoint": str(Path(_checkpoint_label).resolve()),
            "root_priors": "not_generated_by_extraction_schema",
            "cpd_scope": "all_edges" if source == {} else "correctly_matched_edges_only",
        },
    }


def write_pgm_dataset(records, path):
    """Write generated records atomically in PRISM-BN dataset format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "total": len(records),
            "source": str(GROUND_TRUTH_FILE),
            "checkpoint": str(Path(_checkpoint_label).resolve()),
        },
        "bayesian_networks": {
            record["id"]: record for record in records if record and record.get("id")
        },
    }
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temp_path.replace(path)
    print(f"  Generated PGM dataset: {path} ({len(payload['bayesian_networks'])} records)")

# ── Extraction: fine-tuned Qwen ───────────────────────────────────────────────

@torch.no_grad()
def call_qwen(system, user, max_tokens=2000):
    """Extraction calls — fine-tuned Qwen via transformers. Greedy decoding for determinism."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            prompt = _qwen_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = _qwen_tok(prompt, return_tensors="pt").to(_qwen_model.device)
            out = _qwen_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=_qwen_tok.pad_token_id,
                eos_token_id=_qwen_tok.eos_token_id,
                use_cache=True,
            )
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = _qwen_tok.decode(gen, skip_special_tokens=True).strip()
            if _print_raw_output:
                print(f"      [Qwen raw output]: {text}")
            return extract_json_robust(text)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"      [JSON err {attempt+1}]: {e}")
        except Exception as e:
            print(f"      [gen err {attempt+1}]: {e}")
    return None

# ── Judge: Llama via HF InferenceClient ──────────────────────────────────────

def call_llama(hf_client, system, user, max_tokens=2000):
    for attempt in range(MAX_RETRIES):
        try:
            completion = hf_client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = completion.choices[0].message.content.strip()
            print(f"      [Llama raw output]: {text}")
            return extract_json_robust(text)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"      [JSON err {attempt+1}]: {e}"); time.sleep(1)
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                print("      [Rate limit — 60s]"); time.sleep(60)
            else:
                print(f"      [API err {attempt+1}]: {e}"); time.sleep(2)
    return None

# ── Metric helpers ────────────────────────────────────────────────────────────

def f1_prf(n_corr, n_gt, n_pred):
    prec = n_corr / n_pred if n_pred > 0 else 0.0
    rec = n_corr / n_gt if n_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return round(f1, 4), round(prec, 4), round(rec, 4)

def kl_div(p, q):
    p = [0.0 if v is None else float(v) for v in p]
    q = [0.0 if v is None else float(v) for v in q]
    pt = sum(p); p = [v/pt for v in p] if pt > 0 else [1.0/len(p)] * len(p)
    qs = [max(v, 0.0) + LAPLACE_EPS for v in q]; qn = [v/sum(qs) for v in qs]
    return sum(pi * math.log(pi/qi) for pi, qi in zip(p, qn) if pi > 0)

def is_numeric_matrix(matrix):
    """Accept the existing 2-D numeric/nullable format, never flatten ranges.

    Ragged/empty matrices and numeric strings retain the scorer's existing
    handling. Nested cells, non-numbers and non-finite values are unscorable.
    """
    if not isinstance(matrix, list):
        return False
    for row in matrix:
        if not isinstance(row, list):
            return False
        for value in row:
            if value is None:
                continue
            if isinstance(value, (list, dict, bool)):
                return False
            try:
                if not math.isfinite(float(value)):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
    return True


def col_kl(pred_matrix, gt_matrix, n_gt_parent_states, n_gt_child_states):
    if not is_numeric_matrix(pred_matrix) or not is_numeric_matrix(gt_matrix):
        return None
    nr = len(pred_matrix)
    nc = len(pred_matrix[0]) if pred_matrix else 0
    gt_nr = len(gt_matrix)
    kls = []
    for j in range(n_gt_parent_states):
        if j >= nc:
            break
        gt_col = [float(gt_matrix[r][j]) if r < gt_nr and j < len(gt_matrix[r]) and gt_matrix[r][j] is not None else 0.0
                  for r in range(n_gt_child_states)]
        pr_col = [float(pred_matrix[r][j]) if j < len(pred_matrix[r]) and pred_matrix[r][j] is not None else 0.0
                  for r in range(nr)]
        if len(pr_col) > len(gt_col):
            pr_col = pr_col[:len(gt_col)]
        elif len(pr_col) < len(gt_col):
            pr_col += [0.0] * (len(gt_col) - len(pr_col))
        cs = sum(pr_col)
        pr_col = [v/cs for v in pr_col] if cs > 0 else [1.0/len(gt_col)] * len(gt_col)
        kls.append(kl_div(gt_col, pr_col))
    return round(sum(kls)/len(kls), 6) if kls else None

# ── Pass 1 eval ───────────────────────────────────────────────────────────────

def eval_pass1(judge_result, n_gt, n_pred):
    valid = [m for m in judge_result.get("matches", []) if m.get("score", 0) > 0 and "predicted" in m and "ground_truth" in m]

    by_gt = defaultdict(list)
    for m in valid:
        by_gt[m["ground_truth"]].append(m)

    canonical_matches, extras_map = [], {}
    for gt_name, matches in by_gt.items():
        canonical_matches.append(matches[0])
        if len(matches) > 1:
            extras_map[gt_name] = [m["predicted"] for m in matches[1:]]

    n_extra = sum(len(v) for v in extras_map.values())
    n_corr = len(canonical_matches)
    n_pred_eff = max(n_pred - n_extra, n_corr)
    f1, prec, rec = f1_prf(n_corr, n_gt, n_pred_eff)

    metrics = {
        "node_f1": f1, "node_precision": prec, "node_recall": rec,
        "pct_nodes_matched": round(n_corr / n_gt * 100, 1) if n_gt > 0 else 0.0,
        "n_matched_nodes": n_corr, "n_gt_nodes": n_gt, "n_pred_nodes": n_pred,
        "n_extra_nodes": n_extra,
    }
    node_match_map = {m["predicted"]: m["ground_truth"] for m in canonical_matches}
    return metrics, node_match_map, extras_map

# ── Pass 2 eval ───────────────────────────────────────────────────────────────

def eval_pass2_node(judge_result, gt_states, pred_states):
    valid = [m for m in judge_result.get("state_matches", []) if m.get("score", 0) > 0 and "predicted" in m and "ground_truth" in m]
    n_corr = len(valid)
    f1, prec, rec = f1_prf(n_corr, len(gt_states), len(pred_states))
    metrics = {"state_f1": f1, "state_precision": prec, "state_recall": rec,
               "n_matched": n_corr, "n_gt": len(gt_states), "n_pred": len(pred_states)}
    state_map = {m["predicted"]: m["ground_truth"] for m in valid}
    return metrics, state_map

# ── Pass 3 eval ───────────────────────────────────────────────────────────────

def eval_pass3(gt_edges, pred_edges, node_match_map):
    gt_set = {(e["parent"], e["child"]) for e in gt_edges}
    pred_mapped = []
    for e in pred_edges:
        pg = node_match_map.get(e.get("parent", ""))
        cg = node_match_map.get(e.get("child", ""))
        if pg and cg:
            pred_mapped.append((pg, cg))
    pred_set = set(pred_mapped)
    correct = gt_set & pred_set
    spurious = pred_set - gt_set
    n_gt, n_pred, n_corr = len(gt_set), len(pred_set), len(correct)
    f1, prec, rec = f1_prf(n_corr, n_gt, n_pred)
    return {
        "edge_f1": f1, "edge_precision": prec, "edge_recall": rec,
        "pct_correct_edges": round(n_corr/n_gt*100, 1) if n_gt > 0 else 0.0,
        "pct_spurious_edges": round(len(spurious)/n_pred*100, 1) if n_pred > 0 else 0.0,
        "n_correct_edges": n_corr, "n_gt_edges": n_gt, "n_pred_edges": n_pred,
        "n_spurious_edges": len(spurious),
        "correct_edges": list(correct),
    }

# ── Process one subgraph ──────────────────────────────────────────────────────

def process_subgraph(sg_id, sg, llama_client):
    print(f"  [DEBUG] Processing subgraph: {sg_id}")
    result = {
        "id": sg_id, "title": sg.get("title", ""), "domain": sg.get("domain", ""),
        "n_nodes": sg.get("n_nodes", 0), "n_edges": sg.get("n_edges", 0),
        "n_total_states": None, "error": None,
        "node_f1": None, "node_precision": None, "node_recall": None,
        "pct_nodes_matched": None, "n_matched_nodes": None,
        "n_gt_nodes": None, "n_pred_nodes": None, "n_extra_nodes": None,
        "state_f1": None, "state_precision": None, "state_recall": None,
        "pct_states_matched": None,
        "edge_f1": None, "edge_precision": None, "edge_recall": None,
        "pct_correct_edges": None, "pct_spurious_edges": None,
        "n_correct_edges": None, "n_gt_edges": None, "n_pred_edges": None,
        "cpd_kl": None, "cpd_kl_symmetric": None, "n_cpd_evaluated": None,
    }

    text = sg.get("generated_text", "")
    if not text:
        print(f"  [DEBUG] ERROR: no_generated_text")
        result["error"] = "no_generated_text"; return result
    print("TEXT: ", text)
    # print(a)
    gt_nodes = sg.get("nodes", {})
    gt_names = list(gt_nodes.keys())
    result["n_total_states"] = sum(len(info.get("states", [])) for info in gt_nodes.values())
    gt_edges = [{"parent": p, "child": c}
                for c, info in gt_nodes.items()
                for p in info.get("parents", {}).keys()]

    # ── Pass 1: extract nodes (fine-tuned Qwen) ─────────────────────────────
    print("    Pass 1: nodes...")
    p1 = call_qwen(PASS1_SYSTEM, pass1_prompt(text),
                   max_tokens=min(1000, 200 + len(gt_names) * 50))
    print("P1: ", p1)
             
    if not p1:
        result["error"] = "pass1_failed"; return result

    pred_names = p1.get("nodes", [])
    print("PRED NAMES: ", pred_names)
    print("GT NAMES: ", gt_names)
    # print(a)
    if not pred_names:
        result["error"] = "no_nodes_extracted"; return result

    nj = call_llama(llama_client, NODE_JUDGE_SYSTEM,
                    node_judge_prompt(gt_names, pred_names),
                    max_tokens=min(2000, 500 + len(gt_names)*60 + len(pred_names)*60))
    time.sleep(DELAY)
    if not nj:
        result["error"] = "node_judge_failed"; return result

    node_metrics, node_match_map, extras_map = eval_pass1(nj, len(gt_names), len(pred_names))
    result.update(node_metrics)
    print(f"      GT nodes: {gt_names}")
    print(f"      Pred nodes: {pred_names}")
    print(f"      node_f1={node_metrics['node_f1']} "
          f"matched={node_metrics['n_matched_nodes']}/{node_metrics['n_gt_nodes']} "
          f"extras={node_metrics['n_extra_nodes']}")
  
    if not node_match_map:
        result["error"] = "no_node_matches"; return result
    
    # ── Pass 2: extract states (fine-tuned Qwen) ────────────────────────────
    print("    Pass 2: states...")
    matched_pred_nodes = list(node_match_map.keys())
    all_extra_names = [n for ns in extras_map.values() for n in ns]
    all_pred_for_p2 = matched_pred_nodes + all_extra_names

    p2 = call_qwen(PASS2_SYSTEM, pass2_prompt(text, all_pred_for_p2),
                   max_tokens=min(3000, 500 + len(all_pred_for_p2) * 200))

    pred_node_states = {}
    extracted_nodes_raw = {}
    if p2:
        for ns in p2.get("node_states", []):
            states = ns.get("states", [])
            if "None" not in states and "none" not in [s.lower() for s in states]:
                states = states + ["None"]
            extracted_nodes_raw[ns["node"]] = states

        # Match extracted node names to predicted node names
        for extracted_name, states in extracted_nodes_raw.items():
            # Try exact match first
            matched = False
            for pred_name in matched_pred_nodes:
                if extracted_name.lower() == pred_name.lower():
                    pred_node_states[pred_name] = states
                    matched = True
                    break

            # Try matching via ground truth node name
            if not matched:
                for pred_name, gt_name in node_match_map.items():
                    if extracted_name.lower() == gt_name.lower():
                        pred_node_states[pred_name] = states
                        matched = True
                        break

            # If still no match, check for substring matches
            if not matched:
                for pred_name, gt_name in node_match_map.items():
                    if extracted_name.lower() in gt_name.lower() or gt_name.lower() in extracted_name.lower():
                        if pred_name not in pred_node_states or len(states) > len(pred_node_states.get(pred_name, [])):
                            pred_node_states[pred_name] = states
                        matched = True
                        break

    for gt_name, extra_names in extras_map.items():
        canonical = next((p for p, g in node_match_map.items() if g == gt_name), None)
        if canonical is None:
            continue
        combined = list(pred_node_states.get(canonical, []))
        for extra in extra_names:
            for s in pred_node_states.get(extra, []):
                if s not in combined:
                    combined.append(s)
        if "None" not in combined:
            combined.append("None")
        pred_node_states[canonical] = combined

    per_node_state = []
    for pred_name, gt_name in node_match_map.items():
        gt_states = gt_nodes.get(gt_name, {}).get("states", [])
        pred_states = pred_node_states.get(pred_name, [])
        if not gt_states:
            continue
        if not pred_states:
            print(f"        {gt_name}: GT={gt_states}, Pred={pred_states}")
            per_node_state.append({"gt_node": gt_name, "pred_node": pred_name,
                "state_f1": 0.0, "state_precision": 0.0, "state_recall": 0.0,
                "n_matched": 0, "n_gt": len(gt_states), "n_pred": 0})
            continue

        print(f"        {gt_name}: GT={gt_states}, Pred={pred_states}")
        sj = call_llama(llama_client, STATE_JUDGE_SYSTEM,
                        state_judge_prompt(gt_name, pred_name, gt_states, pred_states),
                        max_tokens=min(2000, 400 + len(gt_states)*60 + len(pred_states)*60))
        time.sleep(DELAY)

        if not sj:
            exact = set(s.lower() for s in gt_states) & set(s.lower() for s in pred_states)
            nc = len(exact)
            f1, prec, rec = f1_prf(nc, len(gt_states), len(pred_states))
            m = {"state_f1": f1, "state_precision": prec, "state_recall": rec,
                 "n_matched": nc, "n_gt": len(gt_states), "n_pred": len(pred_states)}
        else:
            m, _ = eval_pass2_node(sj, gt_states, pred_states)

        m["gt_node"] = gt_name; m["pred_node"] = pred_name
        per_node_state.append(m)

    if per_node_state:
        def avg(k):
            v = [d[k] for d in per_node_state if d.get(k) is not None]
            return round(sum(v)/len(v), 4) if v else None
        result["state_f1"] = avg("state_f1")
        result["state_precision"] = avg("state_precision")
        result["state_recall"] = avg("state_recall")
        tot_gt = sum(d["n_gt"] for d in per_node_state)
        tot_mat = sum(d["n_matched"] for d in per_node_state)
        result["pct_states_matched"] = round(tot_mat/tot_gt*100, 1) if tot_gt > 0 else 0.0
        print(f"      state_f1={result['state_f1']} matched={tot_mat}/{tot_gt}")

    # ── Pass 3: extract edges (fine-tuned Qwen) ────────────────────────────
    print("    Pass 3: edges...")
    node_states_for_prompt = []
    for pred_name in matched_pred_nodes:
        states = pred_node_states.get(pred_name, ["None"])
        if "None" not in states and "none" not in [s.lower() for s in states]:
            states = states + ["None"]
        node_states_for_prompt.append({"node": pred_name, "states": states})

    p3 = call_qwen(PASS3_SYSTEM, pass3_prompt(text, node_states_for_prompt),
                   max_tokens=min(2000, 400 + len(matched_pred_nodes) * 100))

    pred_edges = p3.get("edges", []) if p3 else []
    edge_metrics = eval_pass3(gt_edges, pred_edges, node_match_map)
    result.update({k: edge_metrics[k] for k in edge_metrics if k != "correct_edges"})
    print(f"      edge_f1={edge_metrics['edge_f1']} "
          f"correct={edge_metrics['n_correct_edges']}/{edge_metrics['n_gt_edges']} "
          f"spurious={edge_metrics['n_spurious_edges']}")

    # ── Pass 4: CPD matrix for each correctly predicted edge (fine-tuned Qwen) ──
    generated_cpds = []
    generated_prediction = {
        "nodes": node_states_for_prompt,
        "edges": pred_edges,
        "cpds": generated_cpds,
    }
    correct_edges = edge_metrics.get("correct_edges", [])
    if not correct_edges:
        print("      Pass 4: no correct edges — skipping")
        result["generated_pgm"] = make_dataset_record(
            f"{sg_id}_predicted", generated_prediction, text, sg
        )
        return result

    print(f"    Pass 4: CPDs for {len(correct_edges)} correct edges...")
    gt_to_pred = {v: k for k, v in node_match_map.items()}
    kl_values = []
    kl_sym_values = []

    for (gt_parent, gt_child) in correct_edges:
        pred_parent = gt_to_pred.get(gt_parent, gt_parent)
        pred_child = gt_to_pred.get(gt_child, gt_child)

        pred_p_states = pred_node_states.get(pred_parent, [])
        pred_c_states = pred_node_states.get(pred_child, [])
        if not pred_p_states or not pred_c_states:
            continue

        gt_child_info = gt_nodes.get(gt_child, {})
        gt_parent_info = gt_child_info.get("parents", {}).get(gt_parent, {})
        gt_matrix = gt_parent_info.get("cpd_matrix", [])
        gt_p_states = gt_parent_info.get("parent_states", [])
        gt_c_states = gt_child_info.get("states", [])
        if not gt_matrix:
            continue

        p4 = call_qwen(PASS4_SYSTEM,
                       pass4_prompt(text, pred_parent, pred_p_states,
                                    pred_child, pred_c_states),
                       max_tokens=min(2000, 400 + len(pred_p_states)*len(pred_c_states)*20))

        if not p4:
            continue

        generated_cpds.append({
            "parent": pred_parent,
            "child": pred_child,
            "matrix": p4.get("matrix", []),
        })

        if not is_numeric_matrix(p4.get("matrix", [])) or not is_numeric_matrix(gt_matrix):
            source = "prediction" if not is_numeric_matrix(p4.get("matrix", [])) else "ground_truth"
            result.setdefault("cpd_errors", []).append({
                "parent": pred_parent, "child": pred_child,
                "source": source, "error": "invalid_numeric_matrix",
            })
            print(f"        {gt_parent}→{gt_child}: invalid {source} CPD; "
                  "expected 2-D numeric/nullable cells, skipping KL")
            continue

        kl = col_kl(p4.get("matrix", []), gt_matrix, len(gt_p_states), len(gt_c_states))
        kl_rev = col_kl(gt_matrix, p4.get("matrix", []), len(gt_p_states), len(gt_c_states))
        if kl is not None and kl_rev is not None:
            kl_sym = (kl + kl_rev) / 2.0
            kl_values.append(kl)
            kl_sym_values.append(kl_sym)
            print(f"        {gt_parent}→{gt_child} KL={kl:.4f} KL-sym={kl_sym:.4f}")

    if kl_values:
        result["cpd_kl"] = round(sum(kl_values)/len(kl_values), 6)
        result["n_cpd_evaluated"] = len(kl_values)
    if kl_sym_values:
        result["cpd_kl_symmetric"] = round(sum(kl_sym_values)/len(kl_sym_values), 6)
    print(f"      cpd_kl={result['cpd_kl']} cpd_kl_symmetric={result['cpd_kl_symmetric']}")

    result["generated_pgm"] = make_dataset_record(
        f"{sg_id}_predicted", generated_prediction, text, sg
    )

    return result

# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(results):
    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        m = sum(vals)/len(vals)
        return round(m, 4), round(math.sqrt(sum((v-m)**2 for v in vals)/len(vals)), 4)

    keys = ["node_f1", "node_recall", "pct_nodes_matched",
            "state_f1", "state_recall", "pct_states_matched",
            "edge_f1", "edge_recall", "pct_correct_edges", "pct_spurious_edges",
            "cpd_kl", "cpd_kl_symmetric", "n_extra_nodes"]
    agg = {"total": len(results), "extraction_model": BASE_MODEL + f" ({_checkpoint_label})",
           "judge_model": JUDGE_MODEL}
    for k in keys:
        m, s = ms([r.get(k) for r in results])
        agg[f"{k}_mean"] = m; agg[f"{k}_std"] = s
    return agg

# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(results, path):
    fields = ["id", "title", "domain", "n_nodes", "n_edges",
              "node_f1", "node_precision", "node_recall", "pct_nodes_matched",
              "n_matched_nodes", "n_gt_nodes", "n_pred_nodes", "n_extra_nodes",
              "state_f1", "state_precision", "state_recall", "pct_states_matched",
              "edge_f1", "edge_precision", "edge_recall",
              "pct_correct_edges", "pct_spurious_edges",
              "n_correct_edges", "n_gt_edges", "n_pred_edges",
              "cpd_kl", "n_cpd_evaluated", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  CSV: {path}")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _qwen_tok, _qwen_model, _checkpoint_label, _print_raw_output

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=SCRIPT_DIR / "outputs/phase1_q1/final",
                    help="Path to fine-tuned checkpoint")
    direct = ap.add_mutually_exclusive_group()
    direct.add_argument("--text", help="Run inference on this description")
    direct.add_argument("--input-file", type=Path,
                        help="Run inference on a UTF-8 text file")
    ap.add_argument("--prediction-output", type=Path,
                    help="PRISM-format output path for direct inference")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load the base model in 4-bit to reduce GPU memory")
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="Generation limit; Phase 2 outputs often exceed 512 tokens")
    ap.add_argument("--print-raw", action="store_true",
                    help="Print raw model output before parsed JSON")
    ap.add_argument("--output-json", type=Path, default=Path("phase2_eval_result.json"))
    ap.add_argument("--output-csv", type=Path, default=Path("phase2_eval_result.csv"))
    ap.add_argument("--pgm-output", type=Path,
                    help="Generated PGM dataset path; defaults beside --output-json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N subgraphs (for quick testing)")
    args = ap.parse_args()

    if args.max_new_tokens <= 0:
        ap.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: {args.checkpoint} not found"); return
    _checkpoint_label = str(args.checkpoint)
    _print_raw_output = args.print_raw

    _qwen_tok, _qwen_model = load_checkpoint_model(
        args.checkpoint, load_in_4bit=args.load_in_4bit
    )

    # Direct inference deliberately does not initialize the remote judge.
    if args.text is not None or args.input_file is not None:
        if args.text is not None:
            source_text = args.text
        else:
            if not args.input_file.is_file():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
            source_text = args.input_file.read_text(encoding="utf-8")

        prediction = call_qwen(
            FULL_GRAPH_SYSTEM,
            full_graph_prompt(source_text),
            max_tokens=args.max_new_tokens,
        )
        if prediction is None:
            raise RuntimeError(
                "The model did not return valid JSON after all generation attempts"
            )
        print("\nParsed Bayesian network:")
        print(json.dumps(prediction, indent=2))
        prediction_path = args.prediction_output or Path("direct_prediction_pgm.json")
        record = make_dataset_record(
            "direct_prediction", prediction, source_text, source=None
        )
        write_pgm_dataset([record], prediction_path)
        return

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"ERROR: {GROUND_TRUTH_FILE} not found"); return
    if not os.path.exists(TEST_DATA_PATH):
        print(f"ERROR: {TEST_DATA_PATH} not found"); return
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Set HF_TOKEN before running held-out evaluation")
    llama_client = InferenceClient(api_key=hf_token)

    print(f"\n4-Pass BN Evaluation (fine-tuned checkpoint)")
    print(f"  Extraction: {BASE_MODEL} + {args.checkpoint}")
    print(f"  Judge:      {JUDGE_MODEL}")
    print("-" * 65)
    
    results = []
    done_ids = set()
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json) as f:
                ckpt = json.load(f)
            saved_checkpoint = ckpt.get("checkpoint")
            current_checkpoint = str(Path(args.checkpoint).resolve())
            if saved_checkpoint and saved_checkpoint != current_checkpoint:
                raise ValueError(
                    f"output belongs to {saved_checkpoint}; choose a different output path"
                )
            results = ckpt.get("results", [])
            done_ids = {r["id"] for r in results if "id" in r}
            print(f"  [resume] loaded {len(done_ids)} completed samples")
        except Exception as e:
            print(f"  [resume] could not read checkpoint ({e}), starting fresh")

    with open(GROUND_TRUTH_FILE) as f:
        data = json.load(f)
    bns = data.get("bayesian_networks", {})
    print(f"  [DEBUG] Total subgraphs in file: {len(bns)}")

    # Evaluate only held-out subgraph IDs. The old script accidentally evaluated
    # the entire PRISM JSON, including the Phase 2 training set.
    test_dataset = load_from_disk(str(TEST_DATA_PATH))
    test_ids = list(dict.fromkeys(str(value) for value in test_dataset["subgraph_id"]))
    missing_test_ids = [sg_id for sg_id in test_ids if sg_id not in bns]
    if missing_test_ids:
        raise ValueError(
            f"{len(missing_test_ids)} test IDs are absent from {GROUND_TRUTH_FILE}: "
            + ", ".join(missing_test_ids[:5])
        )
    bns = {sg_id: bns[sg_id] for sg_id in test_ids}
    if args.limit:
        bns = dict(list(bns.items())[:args.limit])
    print(f"  [DEBUG] After limit filter: {len(bns)}")
    print(f"  Subgraphs to evaluate: {len(bns)}")

    if args.pgm_output is None:
        args.pgm_output = args.output_json.with_name(
            f"{args.output_json.stem}_pgms.json"
        )
    print(f"  Generated PGM output: {args.pgm_output}")

    total_target = len(bns)
    total_processed = len(done_ids.intersection(bns))
    newly_processed = 0
    run_started = time.monotonic()
    print(f"  [DEBUG] Starting loop with {total_processed} already completed")
    for sg_id, sg in bns.items():
        if sg_id in done_ids:
            print(f"  [DEBUG] Skipping {sg_id} (already done)")
            continue
        print(f"\n[{total_processed+1}/{total_target}] {sg_id} | "
              f"{sg.get('domain','')} | {sg.get('title','')[:40]}")
        r = process_subgraph(sg_id, sg, llama_client)
        results.append(r)
        done_ids.add(sg_id)
        total_processed += 1
        newly_processed += 1
        if r.get("error"):
            print(f"  FAIL — {r['error']}")
        else:
            print(f"  node_f1={r.get('node_f1')} state_f1={r.get('state_f1')} "
                  f"edge_f1={r.get('edge_f1')} cpd_kl={r.get('cpd_kl')} "
                  f"extra_nodes={r.get('n_extra_nodes')}")

        elapsed = time.monotonic() - run_started
        average = elapsed / newly_processed
        remaining = max(0, total_target - total_processed)
        percent = 100.0 * total_processed / max(total_target, 1)
        print(f"  [progress] {total_processed}/{total_target} ({percent:.1f}%) | "
              f"elapsed {format_duration(elapsed)} | "
              f"ETA {format_duration(average * remaining)}")

        pgm_records = [
            item["generated_pgm"]
            for item in results
            if item.get("generated_pgm")
        ]
        write_pgm_dataset(pgm_records, args.pgm_output)

        if (total_processed % 10) == 0:
            print(f"  [checkpoint] saving after {total_processed} samples...")
            ckpt_agg = aggregate(results)
            with open(args.output_json, "w") as f:
                json.dump({"checkpoint": str(Path(args.checkpoint).resolve()),
                           "extraction_model": f"{BASE_MODEL} ({args.checkpoint})",
                           "judge_model": JUDGE_MODEL,
                           "aggregate": ckpt_agg, "results": results}, f, indent=2)
            write_csv(results, args.output_csv)

    if results:
        agg = aggregate(results)
        print(f"\n{'='*65}")
        print(f"  Node F1:         {agg['node_f1_mean']} ± {agg['node_f1_std']}")
        print(f"  % Nodes matched: {agg['pct_nodes_matched_mean']}")
        print(f"  Extra nodes:     {agg['n_extra_nodes_mean']} ± {agg['n_extra_nodes_std']}")
        print(f"  State F1:        {agg['state_f1_mean']} ± {agg['state_f1_std']}")
        print(f"  % States matched:{agg['pct_states_matched_mean']}")
        print(f"  Edge F1:         {agg['edge_f1_mean']} ± {agg['edge_f1_std']}")
        print(f"  % Correct edges: {agg['pct_correct_edges_mean']}")
        print(f"  % Spurious edges:{agg['pct_spurious_edges_mean']}")
        print(f"  CPD-KL:          {agg['cpd_kl_mean']} ± {agg['cpd_kl_std']}")
        print(f"  CPD-KL-Sym:      {agg['cpd_kl_symmetric_mean']} ± {agg['cpd_kl_symmetric_std']}")

        with open(args.output_json, "w") as f:
            json.dump({"checkpoint": str(Path(args.checkpoint).resolve()),
                       "extraction_model": f"{BASE_MODEL} ({args.checkpoint})",
                       "judge_model": JUDGE_MODEL,
                       "aggregate": agg, "results": results}, f, indent=2)
        print(f"  JSON: {args.output_json}")
        write_csv(results, args.output_csv)
        pgm_records = [
            item["generated_pgm"]
            for item in results
            if item.get("generated_pgm")
        ]
        write_pgm_dataset(pgm_records, args.pgm_output)

if __name__ == "__main__":
    main()
