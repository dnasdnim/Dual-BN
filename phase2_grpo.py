"""
DUAL-BN Phase 2: complexity-regularized GRPO fine-tuning for Bayesian
Network extraction.

The reward for a generated graph G_hat is

    R_struct = 0.25 * (F1_node + F1_state + F1_edge + exp(-KL_CPD))
    Z = w_struct + w_cycle + lambda_complexity
    R = clip(
        (w_struct / Z) * R_struct
        + (w_cycle / Z) * R_cycle
        - (lambda_complexity / Z) * P_excess,
        0,
        1,
    )

where P_excess penalizes only graph complexity above the ground-truth graph:
nodes, edges, states, and free CPD parameters. Missing structure receives no
"small graph" bonus; it is already penalized by structural recall and cycle
consistency. The magnitudes of the three final reward coefficients sum to 1.

Smoke test (8 subgraphs, 4 steps):
  export HF_TOKEN=hf_xxx
  CUDA_VISIBLE_DEVICES=0 python phase2_grpo.py --smoke

Full run (2000 steps, default complexity regularization):
  export HF_TOKEN=hf_xxx
  CUDA_VISIBLE_DEVICES=0 python phase2_grpo.py \
      --max-steps 2000 \
      --lambda-complexity 0.05 \
      --output-dir outputs/phase2_grpo_complexity

Complexity ablation:
  CUDA_VISIBLE_DEVICES=0 python phase2_grpo.py \
      --max-steps 2000 \
      --lambda-complexity 0.0 \
      --output-dir outputs/phase2_grpo_no_complexity

Cycle-reward ablation:
  CUDA_VISIBLE_DEVICES=0 python phase2_grpo.py \
      --max-steps 200 \
      --w-struct 1.0 \
      --w-cycle 0.0 \
      --output-dir evaluations_outputs_no_cycle
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from datasets import Dataset, load_from_disk
from huggingface_hub import InferenceClient
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEFAULT_ADAPTER_PATH = "outputs/phase1_q1/final"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct:groq"
PRISM_JSON = "prism_bn.json"

# Node / state / edge / CPD weights within R_struct.
NODE_WEIGHT = 0.25
STATE_WEIGHT = 0.25
EDGE_WEIGHT = 0.25
CPD_WEIGHT = 0.25

DEFAULT_COMPLEXITY_WEIGHTS = {
    "nodes": 0.15,
    "edges": 0.35,
    "states": 0.15,
    "parameters": 0.35,
}
DEFAULT_LAMBDA_COMPLEXITY = 0.05
DEFAULT_COMPLEXITY_CLIP = 1.0

LAPLACE_EPS = 1e-6
MAX_RETRIES = 3


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

EXTRACT_SYSTEM = """You are an expert at extracting Bayesian Networks from natural language.
Given a text, extract the full Bayesian network as JSON: nodes (with states), directed
edges, and for each edge a CPD matrix (rows = child states, columns = parent states,
each column sums to 1.0). Return ONLY valid JSON, no explanation."""


def extract_prompt(text: str) -> str:
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


GEN_SYSTEM = """You are given a parameterized Bayesian Network. Generate a natural-language
description (~350 words) mentioning every variable, every non-"none" state, every causal
relationship, and conveying the STRENGTH of each relationship based on the probability
magnitudes (strong vs weak influence)."""


def gen_prompt(pgm_text: str) -> str:
    return f"""Given the following Bayesian Network, write a faithful natural-language description.

{pgm_text}

Mention every variable, its states, every causal edge, and how strong each influence is."""


NODE_JUDGE_SYSTEM = """Match predicted Bayesian Network node names to ground truth node names.
Scoring: 1=exact match or very minor variation, 0=no meaningful match.
One-to-one matching only. Return ONLY valid JSON."""


def node_judge_prompt(gt_names: Sequence[str], pred_names: Sequence[str]) -> str:
    return f"""Match each predicted node name to a ground truth node name.

Ground truth: {json.dumps(list(gt_names))}
Predicted:    {json.dumps(list(pred_names))}

Return ONLY:
{{"matches": [{{"predicted": "p", "ground_truth": "g", "score": 1}}]}}

Rules:
- score=1 if same meaning or clear paraphrase, else omit
- one predicted maps to at most one ground truth"""


STATE_JUDGE_SYSTEM = """Match predicted Bayesian Network state names to ground truth state names.
Scoring: 1=exact match, minor variation, or clear semantic equivalent; 0=no meaningful match.
Match states only within the supplied matched node pair. Use one-to-one matching.
Return ONLY valid JSON."""


def state_judge_prompt(node_pairs: Sequence[Mapping[str, Any]]) -> str:
    return f"""Match predicted states to ground truth states within each matched node pair.

Node pairs:
{json.dumps(list(node_pairs))}

Return ONLY:
{{"matches": [{{"predicted_node": "p_node",
                "ground_truth_node": "g_node",
                "predicted_state": "p_state",
                "ground_truth_state": "g_state",
                "score": 1}}]}}

Rules:
- match states only within their supplied node pair
- score=1 for the same state meaning or a clear paraphrase, else omit
- one predicted state maps to at most one ground-truth state
- one ground-truth state maps to at most one predicted state"""


CYCLE_JUDGE_SYSTEM = """You evaluate whether a regenerated description faithfully encodes the
same Bayesian network as an original description. Score 1 to 5. Return ONLY valid JSON."""


def cycle_judge_prompt(x: str, x_hat: str) -> str:
    return f"""Original description:
\"\"\"{x}\"\"\"

Regenerated description:
\"\"\"{x_hat}\"\"\"

Score how faithfully the regenerated description captures the SAME Bayesian network as the
original, considering:
- Same variables (nodes) mentioned?
- Same states/values for each variable?
- Same causal relationships (which variable influences which)?
- Is the STRENGTH of each relationship conveyed consistently (strong vs weak, matching the
  probability magnitudes)?

Scoring:
5 = fully faithful (variables, states, edges, and strengths all match)
4 = mostly faithful (minor omissions)
3 = partially faithful (some missing or strengths off)
2 = largely unfaithful (major structure missing)
1 = unfaithful (different/unrecognizable network)

Return ONLY: {{"score": <1-5>, "reason": "<one short sentence>"}}"""


# -----------------------------------------------------------------------------
# JSON and API helpers
# -----------------------------------------------------------------------------

def completion_to_text(completion: Any) -> str:
    """Support both plain-text and conversational TRL completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content", ""))
    if isinstance(completion, Sequence):
        parts = []
        for item in completion:
            if isinstance(item, Mapping):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def extract_json_robust(raw: Any) -> Dict[str, Any]:
    """Extract the first valid JSON object, respecting braces inside strings."""
    text = completion_to_text(raw).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("No valid JSON object found")


def call_judge(
    client: InferenceClient,
    system: str,
    user: str,
    max_tokens: int = 1500,
) -> Dict[str, Any] | None:
    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return extract_json_robust(response.choices[0].message.content.strip())
        except Exception:
            continue
    return None


# -----------------------------------------------------------------------------
# Structural metrics
# -----------------------------------------------------------------------------

def f1(n_correct: int, n_gt: int, n_pred: int) -> float:
    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_gt if n_gt else 0.0
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def kl_div(p: Sequence[Any], q: Sequence[Any]) -> float:
    if not p or len(p) != len(q):
        raise ValueError("KL inputs must be non-empty and have the same length")

    p_values = [_safe_float(value) for value in p]
    q_values = [_safe_float(value) for value in q]

    p_total = sum(p_values)
    if p_total > 0:
        p_norm = [value / p_total for value in p_values]
    else:
        p_norm = [1.0 / len(p_values)] * len(p_values)

    q_smoothed = [max(value, 0.0) + LAPLACE_EPS for value in q_values]
    q_total = sum(q_smoothed)
    q_norm = [value / q_total for value in q_smoothed]

    return sum(
        p_i * math.log(p_i / q_i)
        for p_i, q_i in zip(p_norm, q_norm)
        if p_i > 0
    )


def serialize_pred_pgm(pred: Mapping[str, Any]) -> str:
    lines = ["Nodes and states:"]
    for node in pred.get("nodes", []):
        if isinstance(node, Mapping):
            lines.append(f'  "{node.get("node")}": {node.get("states", [])}')

    lines.append("Edges and CPDs:")
    cpd_lookup = {}
    for cpd in pred.get("cpds", []):
        if isinstance(cpd, Mapping):
            key = (cpd.get("parent"), cpd.get("child"))
            cpd_lookup[key] = cpd.get("matrix")

    for edge in pred.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        parent = edge.get("parent")
        child = edge.get("child")
        matrix = cpd_lookup.get((parent, child), "n/a")
        lines.append(f'  "{parent}"->"{child}"  cpd={matrix}')

    return "\n".join(lines)


def _ordered_unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _validated_node_map(
    judge_result: Mapping[str, Any] | None,
    pred_names: Sequence[str],
    gt_names: Sequence[str],
) -> Dict[str, str]:
    """Enforce the judge prompt's one-to-one mapping constraint locally."""
    if not judge_result:
        return {}

    valid_pred = set(pred_names)
    valid_gt = set(gt_names)
    used_pred = set()
    used_gt = set()
    node_map: Dict[str, str] = {}

    for match in judge_result.get("matches", []):
        if not isinstance(match, Mapping) or match.get("score", 0) <= 0:
            continue
        pred_name = match.get("predicted")
        gt_name = match.get("ground_truth")
        if (
            pred_name not in valid_pred
            or gt_name not in valid_gt
            or pred_name in used_pred
            or gt_name in used_gt
        ):
            continue
        node_map[pred_name] = gt_name
        used_pred.add(pred_name)
        used_gt.add(gt_name)

    return node_map


def _states_by_predicted_node(pred_nodes: Sequence[Any]) -> Dict[str, List[str]]:
    """Collect ordered unique states for every unique predicted node."""
    states_by_node: Dict[str, List[str]] = {}
    for node in pred_nodes:
        if not isinstance(node, Mapping) or not node.get("node"):
            continue
        node_name = str(node["node"])
        states_by_node.setdefault(node_name, [])
        states = node.get("states", [])
        if not isinstance(states, list):
            continue
        for state in states:
            state_name = str(state)
            if state_name not in states_by_node[node_name]:
                states_by_node[node_name].append(state_name)
    return states_by_node


def _states_by_ground_truth_node(
    gt_nodes: Mapping[str, Any],
) -> Dict[str, List[str]]:
    states_by_node: Dict[str, List[str]] = {}
    for node_name, info in gt_nodes.items():
        states = info.get("states", []) if isinstance(info, Mapping) else []
        states_by_node[str(node_name)] = (
            _ordered_unique(str(state) for state in states)
            if isinstance(states, list)
            else []
        )
    return states_by_node


def _state_judge_pairs(
    node_map: Mapping[str, str],
    pred_states: Mapping[str, Sequence[str]],
    gt_states: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    pairs = []
    for pred_node, gt_node in node_map.items():
        predicted = list(pred_states.get(pred_node, []))
        ground_truth = list(gt_states.get(gt_node, []))
        if predicted and ground_truth:
            pairs.append(
                {
                    "predicted_node": pred_node,
                    "ground_truth_node": gt_node,
                    "predicted_states": predicted,
                    "ground_truth_states": ground_truth,
                }
            )
    return pairs


def _validated_state_matches(
    judge_result: Mapping[str, Any] | None,
    node_map: Mapping[str, str],
    pred_states: Mapping[str, Sequence[str]],
    gt_states: Mapping[str, Sequence[str]],
) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Validate one-to-one state matches within the aligned node pairs."""
    if not judge_result:
        return {}

    used_pred: set[Tuple[str, str]] = set()
    used_gt: set[Tuple[str, str]] = set()
    matches: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for match in judge_result.get("matches", []):
        if not isinstance(match, Mapping) or match.get("score", 0) <= 0:
            continue
        pred_node = match.get("predicted_node")
        gt_node = match.get("ground_truth_node")
        pred_state = match.get("predicted_state")
        gt_state = match.get("ground_truth_state")
        if not all(
            isinstance(value, str)
            for value in (pred_node, gt_node, pred_state, gt_state)
        ):
            continue
        pred_key = (pred_node, pred_state)
        gt_key = (gt_node, gt_state)
        if (
            node_map.get(pred_node) != gt_node
            or pred_state not in pred_states.get(pred_node, [])
            or gt_state not in gt_states.get(gt_node, [])
            or pred_key in used_pred
            or gt_key in used_gt
        ):
            continue
        matches[pred_key] = gt_key
        used_pred.add(pred_key)
        used_gt.add(gt_key)
    return matches


def _normalized_state_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _exact_state_matches(
    node_map: Mapping[str, str],
    pred_states: Mapping[str, Sequence[str]],
    gt_states: Mapping[str, Sequence[str]],
) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Match normalized-exact states before requesting semantic alignment."""
    matches: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for pred_node, gt_node in node_map.items():
        gt_lookup = {
            _normalized_state_name(gt_state): gt_state
            for gt_state in gt_states.get(gt_node, [])
        }
        used_gt = set()
        for pred_state in pred_states.get(pred_node, []):
            gt_state = gt_lookup.get(_normalized_state_name(pred_state))
            if gt_state is None or gt_state in used_gt:
                continue
            matches[(pred_node, pred_state)] = (gt_node, gt_state)
            used_gt.add(gt_state)
    return matches


def _rectangular_matrix_shape(matrix: Any) -> Tuple[int, int]:
    """Return rows and common columns, or (0, 0) for a malformed matrix."""
    if not isinstance(matrix, list) or not matrix:
        return 0, 0
    if not all(isinstance(row, list) for row in matrix):
        return 0, 0
    column_counts = [len(row) for row in matrix]
    if not column_counts or min(column_counts) <= 0:
        return 0, 0
    return len(matrix), min(column_counts)


def _mean_cpd_kl(
    correct_edges: Iterable[Tuple[str, str]],
    gt_nodes: Mapping[str, Any],
    pred_cpds: Sequence[Any],
    gt_to_pred: Mapping[str, str],
) -> float:
    """Match the original behavior: average valid columns; otherwise return 5."""
    pred_lookup = {}
    for cpd in pred_cpds:
        if isinstance(cpd, Mapping):
            pred_lookup[(cpd.get("parent"), cpd.get("child"))] = cpd.get("matrix")

    kls: List[float] = []
    for gt_parent, gt_child in correct_edges:
        child_info = gt_nodes.get(gt_child, {})
        parent_info = child_info.get("parents", {}).get(gt_parent, {})
        gt_matrix = parent_info.get("cpd_matrix", [])
        gt_rows, gt_cols = _rectangular_matrix_shape(gt_matrix)
        if not gt_rows or not gt_cols:
            continue

        pred_parent = gt_to_pred.get(gt_parent)
        pred_child = gt_to_pred.get(gt_child)
        pred_matrix = pred_lookup.get((pred_parent, pred_child))
        pred_rows, pred_cols = _rectangular_matrix_shape(pred_matrix)
        if pred_rows < gt_rows or pred_cols < gt_cols:
            continue

        for column in range(gt_cols):
            try:
                gt_column = [
                    _safe_float(gt_matrix[row][column]) for row in range(gt_rows)
                ]
                pred_column = [
                    _safe_float(pred_matrix[row][column]) for row in range(gt_rows)
                ]
                pred_total = sum(pred_column)
                if pred_total > 0:
                    pred_column = [value / pred_total for value in pred_column]
                else:
                    pred_column = [1.0 / gt_rows] * gt_rows
                kls.append(kl_div(gt_column, pred_column))
            except (TypeError, ValueError, OverflowError):
                continue

    return sum(kls) / len(kls) if kls else 5.0


# -----------------------------------------------------------------------------
# Graph-complexity regularizer
# -----------------------------------------------------------------------------

ComplexityCounts = Dict[str, Tuple[int, int]]


def _predicted_state_count(pred_nodes: Sequence[Any]) -> int:
    """Count unique states per unique predicted node."""
    states_by_node: Dict[str, set[str]] = {}
    for node in pred_nodes:
        if not isinstance(node, Mapping) or not node.get("node"):
            continue
        node_name = str(node["node"])
        states_by_node.setdefault(node_name, set())
        states = node.get("states", [])
        if isinstance(states, list):
            states_by_node[node_name].update(str(state) for state in states)
    return sum(len(states) for states in states_by_node.values())


def _ground_truth_state_count(gt_nodes: Mapping[str, Any]) -> int:
    total = 0
    for info in gt_nodes.values():
        states = info.get("states", []) if isinstance(info, Mapping) else []
        if isinstance(states, list):
            total += len({str(state) for state in states})
    return total


def _predicted_parameter_count(pred_cpds: Sequence[Any]) -> int:
    """Count pairwise free parameters: (child rows - 1) * parent columns."""
    total = 0
    for cpd in pred_cpds:
        if not isinstance(cpd, Mapping):
            continue
        rows, columns = _rectangular_matrix_shape(cpd.get("matrix", []))
        total += max(rows - 1, 0) * columns
    return total


def _ground_truth_parameter_count(gt_nodes: Mapping[str, Any]) -> int:
    """Count free parameters in the ground-truth pairwise CPD matrices."""
    total = 0
    for info in gt_nodes.values():
        if not isinstance(info, Mapping):
            continue
        parents = info.get("parents", {})
        if not isinstance(parents, Mapping):
            continue
        for parent_info in parents.values():
            if not isinstance(parent_info, Mapping):
                continue
            rows, columns = _rectangular_matrix_shape(
                parent_info.get("cpd_matrix", [])
            )
            total += max(rows - 1, 0) * columns
    return total


def graph_complexity(
    pred: Mapping[str, Any],
    gt_nodes: Mapping[str, Any],
) -> ComplexityCounts:
    """Return (predicted, ground-truth) counts for N, E, S, and K."""
    pred_nodes = pred.get("nodes", [])
    pred_edges = pred.get("edges", [])
    pred_cpds = pred.get("cpds", [])

    if not isinstance(pred_nodes, list):
        pred_nodes = []
    if not isinstance(pred_edges, list):
        pred_edges = []
    if not isinstance(pred_cpds, list):
        pred_cpds = []

    pred_node_names = {
        str(node.get("node"))
        for node in pred_nodes
        if isinstance(node, Mapping) and node.get("node")
    }
    pred_edge_set = {
        (str(edge.get("parent")), str(edge.get("child")))
        for edge in pred_edges
        if isinstance(edge, Mapping)
        and edge.get("parent")
        and edge.get("child")
    }

    gt_edge_count = 0
    for info in gt_nodes.values():
        if isinstance(info, Mapping) and isinstance(info.get("parents", {}), Mapping):
            gt_edge_count += len(info.get("parents", {}))

    return {
        "nodes": (len(pred_node_names), len(gt_nodes)),
        "edges": (len(pred_edge_set), gt_edge_count),
        "states": (
            _predicted_state_count(pred_nodes),
            _ground_truth_state_count(gt_nodes),
        ),
        "parameters": (
            _predicted_parameter_count(pred_cpds),
            _ground_truth_parameter_count(gt_nodes),
        ),
    }


def relative_excess(pred_count: int, gt_count: int) -> float:
    """Normalized positive surplus; smaller-than-GT counts receive no bonus."""
    return max(0.0, (pred_count - gt_count) / max(gt_count, 1))


def excess_complexity_penalty(
    pred: Mapping[str, Any],
    gt_nodes: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    clip: float | None = DEFAULT_COMPLEXITY_CLIP,
) -> Tuple[float, ComplexityCounts]:
    """
    Compute weighted excess complexity.

    P_excess = sum_m w_m * max(0, (m_pred - m_gt) / max(m_gt, 1))

    Args:
        pred: Parsed predicted graph.
        gt_nodes: Ground-truth node dictionary.
        weights: Weights for nodes, edges, states, and parameters.
        clip: Optional upper bound. Set to None or <= 0 to disable clipping.
    """
    active_weights = dict(weights or DEFAULT_COMPLEXITY_WEIGHTS)
    counts = graph_complexity(pred, gt_nodes)

    penalty = sum(
        active_weights[name] * relative_excess(pred_count, gt_count)
        for name, (pred_count, gt_count) in counts.items()
    )
    if clip is not None and clip > 0:
        penalty = min(penalty, clip)
    return penalty, counts


def _format_complexity_counts(counts: ComplexityCounts) -> str:
    labels = {"nodes": "N", "edges": "E", "states": "S", "parameters": "K"}
    return " ".join(
        f"{labels[name]}={pred_count}/{gt_count}"
        for name, (pred_count, gt_count) in counts.items()
    )


# -----------------------------------------------------------------------------
# GRPO reward function
# -----------------------------------------------------------------------------

# Global state used by TRL's reward callback.
_judge_client: InferenceClient | None = None
_model = None
_tokenizer = None
_w_struct = 0.5
_w_cycle = 0.5
_lambda_complexity = DEFAULT_LAMBDA_COMPLEXITY
_complexity_weights = dict(DEFAULT_COMPLEXITY_WEIGHTS)
_complexity_clip: float | None = DEFAULT_COMPLEXITY_CLIP
_verbose_rewards = False


def compute_reward(completion: Any, gt_nodes: Mapping[str, Any], text: str) -> float:
    """Compute one complexity-regularized rollout reward in [0, 1]."""
    try:
        pred = extract_json_robust(completion)
    except (TypeError, ValueError, json.JSONDecodeError):
        if _verbose_rewards:
            print("      [reward] invalid JSON -> R=0.000")
        return 0.0

    if _judge_client is None:
        raise RuntimeError("Judge client is not initialized")

    gt_names = list(gt_nodes.keys())
    gt_edges = {
        (parent, child)
        for child, info in gt_nodes.items()
        if isinstance(info, Mapping)
        for parent in info.get("parents", {})
    }

    raw_pred_nodes = pred.get("nodes", [])
    raw_pred_edges = pred.get("edges", [])
    pred_cpds = pred.get("cpds", [])
    if not isinstance(raw_pred_nodes, list):
        raw_pred_nodes = []
    if not isinstance(raw_pred_edges, list):
        raw_pred_edges = []
    if not isinstance(pred_cpds, list):
        pred_cpds = []

    pred_names = _ordered_unique(
        str(node.get("node"))
        for node in raw_pred_nodes
        if isinstance(node, Mapping) and node.get("node")
    )

    # 1. Node F1 via semantic judge alignment.
    node_judgment = call_judge(
        _judge_client,
        NODE_JUDGE_SYSTEM,
        node_judge_prompt(gt_names, pred_names),
    )
    node_map = _validated_node_map(node_judgment, pred_names, gt_names)
    f1_node = f1(len(node_map), len(gt_names), len(pred_names))

    # 2. State F1 within the aligned nodes. Denominators include every state in
    # the predicted and ground-truth graphs, so unmatched nodes and extra states
    # are penalized rather than silently ignored.
    pred_states = _states_by_predicted_node(raw_pred_nodes)
    gt_states = _states_by_ground_truth_node(gt_nodes)
    state_matches = _exact_state_matches(node_map, pred_states, gt_states)
    exactly_matched_pred = set(state_matches)
    exactly_matched_gt = set(state_matches.values())
    remaining_pred_states = {
        pred_node: [
            state
            for state in states
            if (pred_node, state) not in exactly_matched_pred
        ]
        for pred_node, states in pred_states.items()
    }
    remaining_gt_states = {
        gt_node: [
            state
            for state in states
            if (gt_node, state) not in exactly_matched_gt
        ]
        for gt_node, states in gt_states.items()
    }
    state_pairs = _state_judge_pairs(
        node_map,
        remaining_pred_states,
        remaining_gt_states,
    )
    if state_pairs:
        state_judgment = call_judge(
            _judge_client,
            STATE_JUDGE_SYSTEM,
            state_judge_prompt(state_pairs),
            max_tokens=2000,
        )
        semantic_matches = _validated_state_matches(
            state_judgment,
            node_map,
            remaining_pred_states,
            remaining_gt_states,
        )
        state_matches.update(semantic_matches)
    n_pred_states = sum(len(states) for states in pred_states.values())
    n_gt_states = sum(len(states) for states in gt_states.values())
    f1_state = f1(len(state_matches), n_gt_states, n_pred_states)

    # 3. Edge F1 after translating predicted node names through the alignment.
    mapped_pred_edges = set()
    for edge in raw_pred_edges:
        if not isinstance(edge, Mapping):
            continue
        parent = node_map.get(str(edge.get("parent", "")))
        child = node_map.get(str(edge.get("child", "")))
        if parent and child:
            mapped_pred_edges.add((parent, child))

    correct_edges = gt_edges & mapped_pred_edges
    f1_edge = f1(len(correct_edges), len(gt_edges), len(mapped_pred_edges))

    # 4. CPD fidelity on correctly recovered edges.
    gt_to_pred = {gt_name: pred_name for pred_name, gt_name in node_map.items()}
    cpd_kl = _mean_cpd_kl(correct_edges, gt_nodes, pred_cpds, gt_to_pred)
    cpd_term = math.exp(-cpd_kl)
    r_struct = (
        NODE_WEIGHT * f1_node
        + STATE_WEIGHT * f1_state
        + EDGE_WEIGHT * f1_edge
        + CPD_WEIGHT * cpd_term
    )

    # 5. Cycle consistency. Skip the expensive generation in a no-cycle ablation.
    cycle_score = 1
    r_cycle = 0.0
    if _w_cycle > 0:
        if _model is None or _tokenizer is None:
            raise RuntimeError("Model and tokenizer are not initialized")

        pred_pgm_text = serialize_pred_pgm(pred)
        with torch.no_grad():
            messages = [
                {"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": gen_prompt(pred_pgm_text)},
            ]
            prompt = _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
            output = _model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=_tokenizer.pad_token_id,
            )
            input_length = inputs["input_ids"].shape[1]
            x_hat = _tokenizer.decode(
                output[0][input_length:],
                skip_special_tokens=True,
            ).strip()

        cycle_judgment = call_judge(
            _judge_client,
            CYCLE_JUDGE_SYSTEM,
            cycle_judge_prompt(text, x_hat),
        )
        raw_score = cycle_judgment.get("score", 1) if cycle_judgment else 1
        try:
            cycle_score = max(1, min(5, int(raw_score)))
        except (TypeError, ValueError):
            cycle_score = 1
        r_cycle = (cycle_score - 1) / 4.0

    # 6. Excess-complexity regularization.
    complexity_penalty, complexity_counts = excess_complexity_penalty(
        pred,
        gt_nodes,
        weights=_complexity_weights,
        clip=_complexity_clip,
    )

    base_reward = _w_struct * r_struct + _w_cycle * r_cycle
    reward = base_reward - _lambda_complexity * complexity_penalty
    reward = min(1.0, max(0.0, reward))

    if _verbose_rewards:
        print(
            "      [reward] "
            f"node_f1={f1_node:.3f} state_f1={f1_state:.3f} "
            f"edge_f1={f1_edge:.3f} "
            f"cpd_kl={cpd_kl:.3f} r_struct={r_struct:.3f} | "
            f"cycle={cycle_score} r_cycle={r_cycle:.3f} | "
            f"{_format_complexity_counts(complexity_counts)} "
            f"p_excess={complexity_penalty:.3f} "
            f"base={base_reward:.3f} R={reward:.3f}"
        )

    return reward


def reward_fn(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """TRL reward callback; dataset columns arrive through kwargs."""
    del prompts  # The source text is passed explicitly as generated_text.

    gt_nodes_jsons = kwargs.get("gt_nodes_json", [])
    texts = kwargs.get("generated_text", [])
    if not (
        len(completions) == len(gt_nodes_jsons) == len(texts)
    ):
        raise ValueError(
            "Reward inputs have different lengths: "
            f"completions={len(completions)}, "
            f"gt_nodes_json={len(gt_nodes_jsons)}, texts={len(texts)}"
        )

    rewards = []
    for completion, gt_nodes_json, text in zip(
        completions,
        gt_nodes_jsons,
        texts,
    ):
        gt_nodes = json.loads(gt_nodes_json)
        rewards.append(compute_reward(completion, gt_nodes, text))
    return rewards


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------

def build_grpo_dataset(
    prism_data: Mapping[str, Any],
    split_subgraph_ids: Sequence[str],
    tokenizer: Any,
    limit: int | None = None,
) -> Dataset:
    """Build one GRPO example per PRISM-BN subgraph."""
    examples = []
    selected_ids = split_subgraph_ids[:limit]

    for subgraph_id in selected_ids:
        subgraph = prism_data.get(subgraph_id)
        if (
            not subgraph
            or "generated_text" not in subgraph
            or "nodes" not in subgraph
        ):
            continue

        text = subgraph["generated_text"]
        gt_nodes = subgraph["nodes"]
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": extract_prompt(text)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        examples.append(
            {
                "prompt": prompt_text,
                "gt_nodes_json": json.dumps(gt_nodes),
                "generated_text": text,
                "subgraph_id": subgraph_id,
            }
        )

    if not examples:
        raise ValueError("No valid GRPO examples were constructed")

    return Dataset.from_dict(
        {key: [example[key] for example in examples] for key in examples[0]}
    )


# -----------------------------------------------------------------------------
# Command-line configuration
# -----------------------------------------------------------------------------

def _validate_unit_sum(name: str, values: Mapping[str, float]) -> None:
    if any(value < 0 for value in values.values()):
        raise ValueError(f"{name} weights must be non-negative: {values}")
    total = sum(values.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} weights must sum to 1.0, got {total:.6f}")


def _normalize_final_reward_coefficients(
    w_struct: float,
    w_cycle: float,
    lambda_complexity: float,
) -> Tuple[float, float, float]:
    coefficients = {
        "struct": w_struct,
        "cycle": w_cycle,
        "complexity": lambda_complexity,
    }
    if any(value < 0 for value in coefficients.values()):
        raise ValueError(
            f"Final reward coefficients must be non-negative: {coefficients}"
        )

    total = sum(coefficients.values())
    if total <= 0:
        raise ValueError("At least one final reward coefficient must be positive")

    return (
        w_struct / total,
        w_cycle / total,
        lambda_complexity / total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complexity-regularized GRPO for DUAL-BN Phase 2"
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=200)

    parser.add_argument(
        "--w-struct",
        type=float,
        default=1,
        help="Relative structural-reward coefficient; normalized with the other final coefficients",
    )
    parser.add_argument(
        "--w-cycle",
        type=float,
        default=0,
        help="Relative cycle-reward coefficient; normalized with the other final coefficients",
    )
    parser.add_argument(
        "--lambda-complexity",
        type=float,
        default=DEFAULT_LAMBDA_COMPLEXITY,
        help=(
            "Relative excess-complexity penalty coefficient; normalized with "
            "the other final coefficients, and 0 disables it"
        ),
    )
    parser.add_argument(
        "--w-complexity-nodes",
        type=float,
        default=DEFAULT_COMPLEXITY_WEIGHTS["nodes"],
    )
    parser.add_argument(
        "--w-complexity-edges",
        type=float,
        default=DEFAULT_COMPLEXITY_WEIGHTS["edges"],
    )
    parser.add_argument(
        "--w-complexity-states",
        type=float,
        default=DEFAULT_COMPLEXITY_WEIGHTS["states"],
    )
    parser.add_argument(
        "--w-complexity-parameters",
        type=float,
        default=DEFAULT_COMPLEXITY_WEIGHTS["parameters"],
    )
    parser.add_argument(
        "--complexity-clip",
        type=float,
        default=DEFAULT_COMPLEXITY_CLIP,
        help="Upper bound for P_excess; set <=0 to disable clipping",
    )

    parser.add_argument("--output-dir", default="evaluations_outputs_normalized_complexity_no_cycle")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--prism-json", default=PRISM_JSON)
    parser.add_argument("--train-dataset", default="data/train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--log-reward-details",
        action="store_true",
        help="Print per-rollout fidelity and complexity components",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 8 subgraphs, 4 steps, no checkpoint saving, verbose rewards",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Dict[str, float]:
    _normalize_final_reward_coefficients(
        args.w_struct,
        args.w_cycle,
        args.lambda_complexity,
    )
    complexity_weights = {
        "nodes": args.w_complexity_nodes,
        "edges": args.w_complexity_edges,
        "states": args.w_complexity_states,
        "parameters": args.w_complexity_parameters,
    }
    _validate_unit_sum("Complexity", complexity_weights)

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be positive")
    if args.save_steps < 0:
        raise ValueError("--save-steps cannot be negative")
    return complexity_weights


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    global _judge_client
    global _model
    global _tokenizer
    global _w_struct
    global _w_cycle
    global _lambda_complexity
    global _complexity_weights
    global _complexity_clip
    global _verbose_rewards

    args = parse_args()
    complexity_weights = validate_args(args)

    if args.smoke:
        args.limit = 8
        args.max_steps = 4
        args.save_steps = 0

    _w_struct, _w_cycle, _lambda_complexity = (
        _normalize_final_reward_coefficients(
            args.w_struct,
            args.w_cycle,
            args.lambda_complexity,
        )
    )
    _complexity_weights = complexity_weights
    _complexity_clip = args.complexity_clip if args.complexity_clip > 0 else None
    _verbose_rewards = args.smoke or args.log_reward_details

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Set HF_TOKEN before starting training")

    print("=" * 78)
    print("DUAL-BN Phase 2: Complexity-Regularized GRPO")
    print(f"  Adapter: {args.adapter_path}")
    print(
        "  Normalized final reward coefficients: "
        f"struct={_w_struct:.6f}, cycle={_w_cycle:.6f}, "
        f"complexity={_lambda_complexity:.6f}"
    )
    print(
        "  Complexity: "
        f"weights={_complexity_weights}, clip={_complexity_clip}"
    )
    print(f"  Max steps: {args.max_steps}, grad accum: {args.grad_accum}")
    print(f"  Output: {args.output_dir}")
    print("=" * 78)

    # Load the Phase 1 policy and make the adapter trainable.
    print("\n[1/4] Loading model and Phase 1 adapter...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )
    _model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        is_trainable=True,
    )
    print(f"  Model loaded. GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    _judge_client = InferenceClient(api_key=hf_token)

    # Build the GRPO dataset.
    print("\n[2/4] Building GRPO dataset...")
    with open(args.prism_json, "r", encoding="utf-8") as handle:
        prism_data = json.load(handle)["bayesian_networks"]
    train_dataset = load_from_disk(args.train_dataset)
    train_subgraph_ids = list(dict.fromkeys(train_dataset["subgraph_id"]))
    if args.limit is not None:
        train_subgraph_ids = train_subgraph_ids[:args.limit]
    print(f"  Train subgraphs: {len(train_subgraph_ids)}")

    grpo_dataset = build_grpo_dataset(
        prism_data,
        train_subgraph_ids,
        _tokenizer,
    )
    print(f"  GRPO examples: {len(grpo_dataset)}")

    # Configure GRPO.
    print("\n[3/4] Setting up GRPOTrainer...")
    save_strategy = "steps" if args.save_steps > 0 else "no"
    config_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        num_generations=4,
        temperature=0.9,
        top_p=0.95,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy=save_strategy,
        save_total_limit=3,
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        seed=42,
    )
    if args.save_steps > 0:
        config_kwargs["save_steps"] = args.save_steps

    grpo_config = GRPOConfig(**config_kwargs)
    trainer = GRPOTrainer(
        model=_model,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=reward_fn,
        processing_class=_tokenizer,
    )

    print("\n[4/4] Starting training...")
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)

    if last_checkpoint is not None:
        print(f"  Resuming from Phase 2 checkpoint: {last_checkpoint}")
    else:
        print("  No Phase 2 checkpoint found; starting from the Phase 1 adapter.")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("\nTraining complete!")
    print(f"Checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
