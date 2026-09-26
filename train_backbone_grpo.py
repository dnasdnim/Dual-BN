"""QLoRA/GRPO training for {all subgraphs in one case} -> one backbone BN.

The module keeps heavyweight ML imports inside ``run_training`` so dataset
validation and unit tests work without a GPU, TRL, or Transformers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRISM_JSON = SCRIPT_DIR / "prism_bn.json"
DEFAULT_BACKBONE_JSONL = SCRIPT_DIR / "bn_marginal_raw.jsonl"
DEFAULT_ADAPTER_PATH = SCRIPT_DIR / "outputs/phase2_grpo_complexity/checkpoint-1200"
DEFAULT_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct:groq"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/backbone_grpo"
DEFAULT_SPLIT_ROOT = SCRIPT_DIR / "data"
LAPLACE_EPS = 1e-6

BACKBONE_SYSTEM = """You are an expert in Bayesian-network consolidation.

You will receive multiple Bayesian-network subgraphs describing parts of a
larger Bayesian network. Produce one canonical backbone Bayesian network.

Identify nodes that represent the same random variable even when their names
are paraphrases. Merge only genuinely equivalent variables, not variables that
are merely related. Harmonize compatible state names, retain unmatched
variables, combine supported directed edges, remove duplicates, resolve
conflicts conservatively, and ensure that the result is a valid directed
acyclic graph.

Preserve all information supported by the input subgraphs. Do not invent
unsupported nodes, states, edges or probabilities. The final graph may contain
disconnected components when the supplied subgraphs do not justify a
connection.

Return only valid JSON with no explanation."""

OUTPUT_INSTRUCTIONS = """Produce exactly this JSON schema:
{
  "nodes": [{"node": "canonical node name", "states": ["state 1", "state 2", "none"]}],
  "edges": [{"parent": "canonical parent node", "child": "canonical child node"}],
  "cpds": [{"parent": "canonical parent node", "child": "canonical child node",
            "matrix": [[0.7, 0.2], [0.3, 0.8]]}]
}

CPD rows are child states and columns are parent states. Every column sums to
one. Every CPD represents an included edge, and all endpoints occur in nodes.
Return only valid JSON."""

NODE_JUDGE_SYSTEM = """Match predicted Bayesian-network node names to ground-truth
node names by meaning. Use one-to-one matching. Return only valid JSON."""
STATE_JUDGE_SYSTEM = """Match state names by meaning only within each supplied
aligned node pair. Use one-to-one matching. Return only valid JSON."""


# ---------------------------------------------------------------------------
# JSON/schema normalization
# ---------------------------------------------------------------------------

def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content", ""))
    if isinstance(completion, Sequence):
        return "\n".join(
            str(item.get("content", "")) if isinstance(item, Mapping) else str(item)
            for item in completion
        )
    return str(completion)


def extract_json_robust(raw: Any) -> Dict[str, Any]:
    text = completion_to_text(raw).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("completion contains no valid JSON object")


def canonical_graph(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Return deterministic list order without changing matrix state order."""
    nodes = sorted(
        [dict(node) for node in graph.get("nodes", []) if isinstance(node, Mapping)],
        key=lambda item: str(item.get("node", "")),
    )
    edges = sorted(
        [dict(edge) for edge in graph.get("edges", []) if isinstance(edge, Mapping)],
        key=lambda item: (str(item.get("parent", "")), str(item.get("child", ""))),
    )
    cpds = sorted(
        [dict(cpd) for cpd in graph.get("cpds", []) if isinstance(cpd, Mapping)],
        key=lambda item: (str(item.get("parent", "")), str(item.get("child", ""))),
    )
    return {"nodes": nodes, "edges": edges, "cpds": cpds}


def normalize_subgraph(record_or_nodes: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert PRISM's node mapping into the model's list-based graph schema."""
    raw_nodes = record_or_nodes.get("nodes", record_or_nodes)
    if not isinstance(raw_nodes, Mapping):
        raise ValueError("PRISM subgraph 'nodes' must be an object")
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    cpds: List[Dict[str, Any]] = []
    for name, info in raw_nodes.items():
        if not isinstance(info, Mapping):
            raise ValueError(f"node {name!r} must be an object")
        nodes.append({"node": str(name), "states": list(info.get("states", []))})
        parents = info.get("parents", {})
        if not isinstance(parents, Mapping):
            raise ValueError(f"node {name!r} parents must be an object")
        for parent, parent_info in parents.items():
            edges.append({"parent": str(parent), "child": str(name)})
            if isinstance(parent_info, Mapping) and "cpd_matrix" in parent_info:
                cpds.append({
                    "parent": str(parent),
                    "child": str(name),
                    "matrix": parent_info["cpd_matrix"],
                })
    return canonical_graph({"nodes": nodes, "edges": edges, "cpds": cpds})


def _joint_to_matrix(
    table: Mapping[str, Any], parent_states: Sequence[str], child_states: Sequence[str]
) -> List[List[float]]:
    """Convert {parent_state: {child_state: joint}} to P(child|parent)."""
    columns: List[List[float]] = []
    for parent_state in parent_states:
        child_values = table.get(parent_state, {})
        if not isinstance(child_values, Mapping):
            raise ValueError(f"joint column {parent_state!r} must be an object")
        values = [float(child_values.get(child_state, 0.0)) for child_state in child_states]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("joint probabilities must be finite and nonnegative")
        total = sum(values)
        if total <= 0:
            raise ValueError(f"joint column {parent_state!r} has zero mass")
        columns.append([value / total for value in values])
    return [[columns[column][row] for column in range(len(columns))]
            for row in range(len(child_states))]


def normalize_backbone(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize bn_marginal_raw.jsonl, deriving pairwise CPDs from joints."""
    raw_nodes = raw.get("nodes", [])
    if isinstance(raw_nodes, Mapping):
        return normalize_subgraph(raw_nodes)
    if not isinstance(raw_nodes, list):
        raise ValueError("backbone nodes must be a list")
    nodes = []
    states_by_node: Dict[str, List[str]] = {}
    for item in raw_nodes:
        if not isinstance(item, Mapping):
            raise ValueError("each backbone node must be an object")
        name = item.get("name", item.get("node"))
        if not isinstance(name, str) or not name:
            raise ValueError("backbone node requires 'name'")
        states = list(item.get("states", []))
        nodes.append({"node": name, "states": states})
        states_by_node[name] = states

    edges = [
        {"parent": str(edge.get("parent", "")), "child": str(edge.get("child", ""))}
        for edge in raw.get("edges", []) if isinstance(edge, Mapping)
    ]
    # Synthetic/already-normalized targets can carry matrices directly.
    if isinstance(raw.get("cpds"), list):
        return canonical_graph({"nodes": nodes, "edges": edges, "cpds": raw["cpds"]})
    joint_lookup = {
        (str(item.get("parent")), str(item.get("child"))): item.get("table", {})
        for item in raw.get("joints", []) if isinstance(item, Mapping)
    }
    cpds = []
    for edge in edges:
        parent, child = edge["parent"], edge["child"]
        table = joint_lookup.get((parent, child))
        if table is None:
            continue
        cpds.append({
            "parent": parent,
            "child": child,
            "matrix": _joint_to_matrix(table, states_by_node[parent], states_by_node[child]),
        })
    return canonical_graph({"nodes": nodes, "edges": edges, "cpds": cpds})


def load_backbones(path: str | Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            source_id = raw.get("id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"{path}:{line_number}: missing string id")
            graph = normalize_backbone(raw)
            if source_id in result and stable_json(result[source_id]) != stable_json(graph):
                raise ValueError(f"conflicting backbones for source id {source_id}")
            result[source_id] = graph
    return result


def load_subgraphs(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("bayesian_networks")
    if not isinstance(records, Mapping):
        raise ValueError("PRISM JSON requires a 'bayesian_networks' object")
    return [dict(value) for value in records.values()]


def group_subgraph_rows(
    rows: Sequence[Mapping[str, Any]],
    backbone_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Group row-per-subgraph data and attach exactly one consistent backbone."""
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = row.get("parent_id", row.get("backbone_id"))
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every subgraph row requires parent_id/backbone_id")
        grouped[case_id].append(row)

    cases = []
    for case_id in sorted(grouped):
        case_rows = grouped[case_id]
        source_ids = {str(row.get("source_id")) for row in case_rows if row.get("source_id")}
        if len(source_ids) > 1:
            raise ValueError(f"case {case_id} has conflicting source_id values: {sorted(source_ids)}")

        targets: List[Dict[str, Any]] = []
        for row in case_rows:
            embedded = row.get("backbone", row.get("ground_truth_backbone"))
            if isinstance(embedded, Mapping):
                targets.append(normalize_backbone(embedded))
        source_id = next(iter(source_ids), None)
        if backbone_by_source is not None:
            if source_id is None:
                raise ValueError(f"case {case_id} has no source_id for backbone join")
            if source_id not in backbone_by_source:
                raise ValueError(f"case {case_id}: no backbone with id {source_id}")
            targets.append(canonical_graph(backbone_by_source[source_id]))
        if not targets:
            raise ValueError(f"case {case_id} has no ground-truth backbone")
        signatures = {stable_json(target) for target in targets}
        if len(signatures) != 1:
            raise ValueError(f"conflicting ground-truth backbones for case {case_id}")

        subgraphs = [normalize_subgraph(row) for row in case_rows]
        cases.append({
            "backbone_id": case_id,
            "source_id": source_id or "",
            "subgraphs": subgraphs,
            "backbone": targets[0],
            "subgraph_ids": [str(row.get("id", index)) for index, row in enumerate(case_rows)],
        })
    return cases


# ---------------------------------------------------------------------------
# Prompts, augmentation, splits, and reporting
# ---------------------------------------------------------------------------

def subgraphs_user_prompt(subgraphs: Sequence[Mapping[str, Any]]) -> str:
    blocks = [
        # Compact JSON preserves the complete graph while substantially reducing
        # the context consumed by repeated subgraph structure.
        f"SUBGRAPH {index}:\n{json.dumps(graph, ensure_ascii=False, separators=(',', ':'))}"
        for index, graph in enumerate(subgraphs, 1)
    ]
    return "Consolidate all of these subgraphs into one backbone.\n\n" + "\n\n".join(blocks) + "\n\n" + OUTPUT_INSTRUCTIONS


def format_prompt(subgraphs: Sequence[Mapping[str, Any]], tokenizer: Any = None) -> str:
    messages = [
        {"role": "system", "content": BACKBONE_SYSTEM},
        {"role": "user", "content": subgraphs_user_prompt(subgraphs)},
    ]
    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>\n{BACKBONE_SYSTEM}\n<|user|>\n{messages[1]['content']}\n<|assistant|>\n"


def augment_cases(
    cases: Sequence[Mapping[str, Any]], permutations_per_case: int, seed: int,
    tokenizer: Any = None,
) -> List[Dict[str, str]]:
    if permutations_per_case <= 0:
        raise ValueError("permutations_per_case must be positive")
    examples: List[Dict[str, str]] = []
    for case_index, case in enumerate(cases):
        subgraphs = list(case["subgraphs"])
        seen: set[Tuple[int, ...]] = set()
        for permutation_index in range(permutations_per_case):
            order = list(range(len(subgraphs)))
            if permutation_index:
                rng = random.Random(seed + case_index * 1_000_003 + permutation_index)
                rng.shuffle(order)
                # Avoid duplicate order when practical, without enumerating K!.
                retries = 0
                while tuple(order) in seen and retries < 20 and len(order) > 1:
                    rng.shuffle(order)
                    retries += 1
            seen.add(tuple(order))
            ordered = [subgraphs[index] for index in order]
            prompt = format_prompt(ordered, tokenizer)
            target = stable_json(canonical_graph(case["backbone"]))
            examples.append({
                "prompt": prompt,
                "backbone_json": target,
                "subgraphs_json": stable_json(ordered),
                "backbone_id": str(case["backbone_id"]),
            })
    return examples


def approximate_token_count(text: str) -> int:
    """Conservative fallback used only when Transformers is unavailable."""
    return math.ceil(len(text.encode("utf-8")) / 3.0)


def prompt_length_report(
    examples: Sequence[Mapping[str, str]], tokenizer: Any = None,
    max_prompt_tokens: int | None = None,
) -> Dict[str, Any]:
    lengths = []
    over = []
    for example in examples:
        if tokenizer is None:
            length = approximate_token_count(example["prompt"])
        else:
            length = len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
        lengths.append(length)
        if max_prompt_tokens is not None and length > max_prompt_tokens:
            over.append(str(example["backbone_id"]))
    ordered = sorted(lengths)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "minimum": min(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "maximum": max(ordered),
        "over_limit_ids": sorted(set(over)),
        "estimated": tokenizer is None,
    }


def filter_overlength(
    examples: Sequence[Dict[str, str]], tokenizer: Any, max_prompt_tokens: int,
    skip: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    report = prompt_length_report(examples, tokenizer, max_prompt_tokens)
    over = set(report["over_limit_ids"])
    if over and not skip:
        preview = ", ".join(sorted(over)[:20])
        raise ValueError(
            f"{len(over)} backbone cases exceed --max-prompt-tokens={max_prompt_tokens}: "
            f"{preview}. Increase the limit or explicitly pass --skip-overlength."
        )
    return [item for item in examples if item["backbone_id"] not in over], report


def deterministic_case_split(case_ids: Sequence[str], seed: int = 42) -> Dict[str, set[str]]:
    ids = sorted(set(case_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train:n_train + n_val]),
        "test": set(ids[n_train + n_val:]),
    }


def check_split_leakage(splits: Mapping[str, Iterable[str]]) -> None:
    owner: Dict[str, str] = {}
    for split, values in splits.items():
        for case_id in set(values):
            if case_id in owner and owner[case_id] != split:
                raise ValueError(f"backbone leakage: {case_id} occurs in {owner[case_id]} and {split}")
            owner[case_id] = split


def read_saved_splits(split_root: str | Path) -> Dict[str, set[str]] | None:
    root = Path(split_root)
    paths = {name: root / name for name in ("train", "val", "test")}
    if not all(path.is_dir() for path in paths.values()):
        return None
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError(
            f"Hugging Face 'datasets' is required to verify existing splits under {root}"
        ) from error
    result = {}
    for name, path in paths.items():
        dataset = load_from_disk(str(path))
        if "backbone_id" not in dataset.column_names:
            raise ValueError(f"saved split {path} has no backbone_id column")
        result[name] = set(dataset["backbone_id"])
    check_split_leakage(result)
    return result


# ---------------------------------------------------------------------------
# Deterministic graph validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    schema_valid: bool
    dag_valid: bool
    errors: List[str]

    @property
    def valid(self) -> bool:
        return self.schema_valid and self.dag_valid


def _matrix_shape(matrix: Any) -> Tuple[int, int]:
    if not isinstance(matrix, list) or not matrix or not all(isinstance(row, list) for row in matrix):
        return (0, 0)
    widths = {len(row) for row in matrix}
    if len(widths) != 1 or next(iter(widths), 0) == 0:
        return (0, 0)
    return (len(matrix), next(iter(widths)))


def is_dag(node_names: Iterable[str], edges: Iterable[Tuple[str, str]]) -> bool:
    nodes = set(node_names)
    adjacency = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for parent, child in set(edges):
        if parent not in nodes or child not in nodes or parent == child:
            return False
        if child not in adjacency[parent]:
            adjacency[parent].add(child)
            indegree[child] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == len(nodes)


def validate_graph(
    graph: Mapping[str, Any], require_cpds: bool = True, tolerance: float = 1e-3,
) -> ValidationResult:
    errors: List[str] = []
    for field in ("nodes", "edges", "cpds"):
        if field not in graph or not isinstance(graph[field], list):
            errors.append(f"{field} must be a list")
    if errors:
        return ValidationResult(False, False, errors)

    names: List[str] = []
    states_by_name: Dict[str, List[Any]] = {}
    for index, node in enumerate(graph["nodes"]):
        if not isinstance(node, Mapping) or not isinstance(node.get("node"), str) or not node["node"]:
            errors.append(f"node {index} has no nonempty string name")
            continue
        name = node["node"]
        names.append(name)
        states = node.get("states")
        if not isinstance(states, list) or not states:
            errors.append(f"node {name!r} has an empty/non-list states field")
            states = []
        elif len({str(value) for value in states}) != len(states):
            errors.append(f"node {name!r} has duplicate states")
        states_by_name[name] = list(states)
    if len(set(names)) != len(names):
        errors.append("node names are not unique")

    name_set = set(names)
    edge_list: List[Tuple[str, str]] = []
    for index, edge in enumerate(graph["edges"]):
        if not isinstance(edge, Mapping):
            errors.append(f"edge {index} is not an object")
            continue
        parent, child = edge.get("parent"), edge.get("child")
        if not isinstance(parent, str) or not isinstance(child, str):
            errors.append(f"edge {index} requires string endpoints")
            continue
        edge_list.append((parent, child))
        if parent not in name_set or child not in name_set:
            errors.append(f"edge {parent!r}->{child!r} references a missing node")
        if parent == child:
            errors.append(f"self-loop at {parent!r}")
    if len(set(edge_list)) != len(edge_list):
        errors.append("duplicate directed edges")

    cpd_counts: Counter[Tuple[str, str]] = Counter()
    for index, cpd in enumerate(graph["cpds"]):
        if not isinstance(cpd, Mapping):
            errors.append(f"CPD {index} is not an object")
            continue
        key = (cpd.get("parent"), cpd.get("child"))
        if not all(isinstance(value, str) for value in key):
            errors.append(f"CPD {index} requires string endpoints")
            continue
        cpd_counts[key] += 1
        if key not in set(edge_list):
            errors.append(f"CPD {key[0]!r}->{key[1]!r} has no represented edge")
            continue
        rows, columns = _matrix_shape(cpd.get("matrix"))
        expected = (len(states_by_name.get(key[1], [])), len(states_by_name.get(key[0], [])))
        if (rows, columns) != expected:
            errors.append(f"CPD {key} shape {(rows, columns)} != {expected}")
            continue
        matrix = cpd["matrix"]
        for row in matrix:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    errors.append(f"CPD {key} contains a non-probability")
                    break
        for column in range(columns):
            try:
                total = sum(float(matrix[row][column]) for row in range(rows))
            except (TypeError, ValueError):
                continue
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
                errors.append(f"CPD {key} column {column} sums to {total:.6f}")
    for edge in set(edge_list):
        count = cpd_counts[edge]
        if require_cpds and count != 1:
            errors.append(f"edge {edge} requires exactly one CPD, found {count}")
        elif count > 1:
            errors.append(f"edge {edge} has duplicate CPDs")

    dag_valid = is_dag(name_set, edge_list)
    if not dag_valid:
        errors.append("graph is not a DAG")
    schema_errors = [error for error in errors if error != "graph is not a DAG"]
    return ValidationResult(not schema_errors, dag_valid, errors)


# ---------------------------------------------------------------------------
# Alignments and structural/CPD metrics
# ---------------------------------------------------------------------------

def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def exact_one_to_one(predicted: Sequence[str], ground_truth: Sequence[str]) -> Dict[str, str]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for value in ground_truth:
        buckets[normalized_name(value)].append(value)
    used: set[str] = set()
    result = {}
    for value in predicted:
        for candidate in buckets.get(normalized_name(value), []):
            if candidate not in used:
                result[value] = candidate
                used.add(candidate)
                break
    return result


def validate_semantic_node_mapping(
    response: Mapping[str, Any] | None, predicted: Sequence[str], ground_truth: Sequence[str],
) -> Dict[str, str]:
    if not response:
        return {}
    valid_pred, valid_gt = set(predicted), set(ground_truth)
    used_pred: set[str] = set()
    used_gt: set[str] = set()
    result = {}
    for item in response.get("matches", []):
        if not isinstance(item, Mapping) or float(item.get("score", 0)) <= 0:
            continue
        pred, gt = item.get("predicted"), item.get("ground_truth")
        if pred in valid_pred and gt in valid_gt and pred not in used_pred and gt not in used_gt:
            result[pred] = gt
            used_pred.add(pred)
            used_gt.add(gt)
    return result


def f1(correct: int, predicted: int, ground_truth: int) -> float:
    if predicted == 0 or ground_truth == 0:
        return 1.0 if predicted == ground_truth == 0 else 0.0
    precision, recall = correct / predicted, correct / ground_truth
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class SemanticAligner:
    def __init__(self, judge: Callable[[str, str], Mapping[str, Any] | None] | None = None):
        self.judge = judge
        self.cache: Dict[str, Mapping[str, Any] | None] = {}

    def _call(self, system: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self.judge is None:
            return None
        prompt = stable_json(payload)
        key = hashlib.sha256((system + prompt).encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.judge(system, prompt)
        return self.cache[key]

    def nodes(self, predicted: Sequence[str], ground_truth: Sequence[str]) -> Dict[str, str]:
        mapping = exact_one_to_one(predicted, ground_truth)
        remaining_pred = [value for value in predicted if value not in mapping]
        used_gt = set(mapping.values())
        remaining_gt = [value for value in ground_truth if value not in used_gt]
        if remaining_pred and remaining_gt:
            response = self._call(NODE_JUDGE_SYSTEM, {
                "predicted": remaining_pred,
                "ground_truth": remaining_gt,
                "output_schema": {"matches": [{"predicted": "p", "ground_truth": "g", "score": 1}]},
            })
            mapping.update(validate_semantic_node_mapping(response, remaining_pred, remaining_gt))
        return mapping

    def states(
        self, node_map: Mapping[str, str], pred_states: Mapping[str, Sequence[str]],
        gt_states: Mapping[str, Sequence[str]],
    ) -> Dict[Tuple[str, str], Tuple[str, str]]:
        result: Dict[Tuple[str, str], Tuple[str, str]] = {}
        unresolved = []
        for pred_node, gt_node in node_map.items():
            local = exact_one_to_one(list(pred_states.get(pred_node, [])), list(gt_states.get(gt_node, [])))
            for pred_state, gt_state in local.items():
                result[(pred_node, pred_state)] = (gt_node, gt_state)
            remain_pred = [x for x in pred_states.get(pred_node, []) if (pred_node, x) not in result]
            used_gt = {value[1] for value in result.values() if value[0] == gt_node}
            remain_gt = [x for x in gt_states.get(gt_node, []) if x not in used_gt]
            if remain_pred and remain_gt:
                unresolved.append({
                    "predicted_node": pred_node, "ground_truth_node": gt_node,
                    "predicted_states": remain_pred, "ground_truth_states": remain_gt,
                })
        if not unresolved:
            return result
        response = self._call(STATE_JUDGE_SYSTEM, {
            "node_pairs": unresolved,
            "output_schema": {"matches": [{
                "predicted_node": "pn", "ground_truth_node": "gn",
                "predicted_state": "ps", "ground_truth_state": "gs", "score": 1,
            }]},
        })
        valid_pairs = {(item["predicted_node"], item["ground_truth_node"]): item for item in unresolved}
        used_pred = set(result)
        used_gt = set(result.values())
        for item in (response or {}).get("matches", []):
            if not isinstance(item, Mapping) or float(item.get("score", 0)) <= 0:
                continue
            pn, gn = item.get("predicted_node"), item.get("ground_truth_node")
            ps, gs = item.get("predicted_state"), item.get("ground_truth_state")
            pair = valid_pairs.get((pn, gn))
            pred_key, gt_key = (pn, ps), (gn, gs)
            if (pair and ps in pair["predicted_states"] and gs in pair["ground_truth_states"]
                    and pred_key not in used_pred and gt_key not in used_gt):
                result[pred_key] = gt_key
                used_pred.add(pred_key)
                used_gt.add(gt_key)
        return result


def _states_by_node(graph: Mapping[str, Any]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for item in graph.get("nodes", []):
        if isinstance(item, Mapping) and isinstance(item.get("node"), str):
            result.setdefault(item["node"], [])
            if isinstance(item.get("states"), list):
                for state in item["states"]:
                    value = str(state)
                    if value not in result[item["node"]]:
                        result[item["node"]].append(value)
    return result


def directed_edge_f1(
    pred_edges: Sequence[Any], gt_edges: Sequence[Any], node_map: Mapping[str, str],
) -> Tuple[float, set[Tuple[str, str]], int]:
    """Score mapped directed edges while retaining malformed/unmappable denominator entries."""
    gt_set = {
        (str(edge.get("parent")), str(edge.get("child")))
        for edge in gt_edges if isinstance(edge, Mapping)
    }
    unique_pred: set[Any] = set()
    mapped: set[Tuple[str, str]] = set()
    for index, edge in enumerate(pred_edges):
        if not isinstance(edge, Mapping):
            unique_pred.add(("__malformed__", stable_json(edge) if isinstance(edge, (list, dict)) else repr(edge)))
            continue
        parent, child = edge.get("parent"), edge.get("child")
        if not isinstance(parent, str) or not isinstance(child, str):
            unique_pred.add(("__malformed__", index, repr(parent), repr(child)))
            continue
        unique_pred.add((parent, child))
        if parent in node_map and child in node_map:
            mapped.add((node_map[parent], node_map[child]))
    correct = mapped & gt_set
    return f1(len(correct), len(unique_pred), len(gt_set)), correct, len(unique_pred)


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p_smooth = [max(float(value), 0.0) + LAPLACE_EPS for value in p]
    q_smooth = [max(float(value), 0.0) + LAPLACE_EPS for value in q]
    p_total, q_total = sum(p_smooth), sum(q_smooth)
    return sum((pv / p_total) * math.log((pv / p_total) / (qv / q_total))
               for pv, qv in zip(p_smooth, q_smooth))


def cpd_fidelity(
    pred: Mapping[str, Any], gt: Mapping[str, Any], correct_gt_edges: Iterable[Tuple[str, str]],
    node_map: Mapping[str, str], state_map: Mapping[Tuple[str, str], Tuple[str, str]],
) -> Tuple[float, float, int]:
    pred_states, gt_states = _states_by_node(pred), _states_by_node(gt)
    gt_to_pred = {gt_name: pred_name for pred_name, gt_name in node_map.items()}
    pred_cpds = {(item.get("parent"), item.get("child")): item.get("matrix")
                 for item in pred.get("cpds", []) if isinstance(item, Mapping)}
    gt_cpds = {(item.get("parent"), item.get("child")): item.get("matrix")
               for item in gt.get("cpds", []) if isinstance(item, Mapping)}
    kls: List[float] = []
    for gt_parent, gt_child in correct_gt_edges:
        pred_parent, pred_child = gt_to_pred.get(gt_parent), gt_to_pred.get(gt_child)
        pm = pred_cpds.get((pred_parent, pred_child))
        gm = gt_cpds.get((gt_parent, gt_child))
        pr, pc = _matrix_shape(pm)
        gr, gc = _matrix_shape(gm)
        if not all((pred_parent, pred_child, pr, pc, gr, gc)):
            continue
        pred_parent_index = {state: i for i, state in enumerate(pred_states[pred_parent])}
        pred_child_index = {state: i for i, state in enumerate(pred_states[pred_child])}
        gt_parent_to_pred = {
            gt_state: pred_state for (node, pred_state), (gt_node, gt_state) in state_map.items()
            if node == pred_parent and gt_node == gt_parent
        }
        gt_child_to_pred = {
            gt_state: pred_state for (node, pred_state), (gt_node, gt_state) in state_map.items()
            if node == pred_child and gt_node == gt_child
        }
        if not all(state in gt_parent_to_pred for state in gt_states[gt_parent]):
            continue
        if not all(state in gt_child_to_pred for state in gt_states[gt_child]):
            continue
        try:
            row_order = [pred_child_index[gt_child_to_pred[state]] for state in gt_states[gt_child]]
            column_order = [pred_parent_index[gt_parent_to_pred[state]] for state in gt_states[gt_parent]]
            for gt_column, pred_column in enumerate(column_order):
                predicted_values = [float(pm[row][pred_column]) for row in row_order]
                ground_values = [float(gm[row][gt_column]) for row in range(gr)]
                total = sum(max(value, 0.0) for value in predicted_values)
                if total <= 0:
                    continue
                predicted_values = [max(value, 0.0) / total for value in predicted_values]
                kls.append(kl_divergence(ground_values, predicted_values))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    if not kls:
        return 0.0, math.inf, 0
    mean_kl = sum(kls) / len(kls)
    return math.exp(-mean_kl), mean_kl, len(kls)


# ---------------------------------------------------------------------------
# Complexity and final reward
# ---------------------------------------------------------------------------

def free_parameter_count(graph: Mapping[str, Any]) -> int:
    total = 0
    for cpd in graph.get("cpds", []):
        if isinstance(cpd, Mapping):
            rows, columns = _matrix_shape(cpd.get("matrix"))
            total += columns * max(rows - 1, 0)
    return total


def graph_counts(graph: Mapping[str, Any]) -> Dict[str, int]:
    states = _states_by_node(graph)
    edges = {(str(item.get("parent")), str(item.get("child")))
             for item in graph.get("edges", []) if isinstance(item, Mapping)}
    return {
        "nodes": len(states), "edges": len(edges),
        "states": sum(len(values) for values in states.values()),
        "parameters": free_parameter_count(graph),
    }


def excess_complexity_penalty(
    pred: Mapping[str, Any], gt: Mapping[str, Any], weights: Mapping[str, float],
) -> Tuple[float, Dict[str, Tuple[int, int]]]:
    pred_counts, gt_counts = graph_counts(pred), graph_counts(gt)
    pairs = {name: (pred_counts[name], gt_counts[name]) for name in weights}
    penalty = sum(
        weights[name] * max(0, predicted - truth) / max(1, truth)
        for name, (predicted, truth) in pairs.items()
    )
    return penalty, pairs


@dataclass
class RewardConfig:
    node_weight: float = 0.25
    state_weight: float = 0.25
    edge_weight: float = 0.25
    cpd_weight: float = 0.25
    lambda_complexity: float = 0.05
    complexity_weights: Mapping[str, float] | None = None
    invalid_penalty: float = 1.0
    invalid_reward_mode: str = "zero"
    require_cpds: bool = True
    log_details: bool = False

    def __post_init__(self) -> None:
        if self.complexity_weights is None:
            self.complexity_weights = {"nodes": .15, "edges": .35, "states": .15, "parameters": .35}


def compute_reward(
    completion: Any, ground_truth: Mapping[str, Any], backbone_id: str = "",
    aligner: SemanticAligner | None = None, config: RewardConfig | None = None,
) -> Tuple[float, Dict[str, Any]]:
    config = config or RewardConfig()
    aligner = aligner or SemanticAligner()
    try:
        pred = extract_json_robust(completion)
    except ValueError:
        details = {"backbone_id": backbone_id, "schema_valid": False, "dag_valid": False,
                   "final_reward": 0.0, "error": "malformed JSON"}
        return 0.0, details

    validation = validate_graph(pred, config.require_cpds)
    pred_states, gt_states = _states_by_node(pred), _states_by_node(ground_truth)
    node_map = aligner.nodes(list(pred_states), list(gt_states))
    state_map = aligner.states(node_map, pred_states, gt_states)
    node_f1 = f1(len(node_map), len(pred_states), len(gt_states))
    state_f1 = f1(len(state_map), sum(map(len, pred_states.values())), sum(map(len, gt_states.values())))
    edge_f1, correct_edges, pred_edge_count = directed_edge_f1(
        pred.get("edges", []) if isinstance(pred.get("edges"), list) else [],
        ground_truth.get("edges", []), node_map,
    )
    cpd_reward, cpd_kl, comparable_columns = cpd_fidelity(
        pred, ground_truth, correct_edges, node_map, state_map
    )
    fidelity = (config.node_weight * node_f1 + config.state_weight * state_f1
                + config.edge_weight * edge_f1 + config.cpd_weight * cpd_reward)
    excess, counts = excess_complexity_penalty(pred, ground_truth, config.complexity_weights or {})
    invalidity = 0.0 if validation.valid else config.invalid_penalty
    reward = fidelity - config.lambda_complexity * excess - invalidity
    reward = min(1.0, max(0.0, reward))
    if not validation.valid and config.invalid_reward_mode == "zero":
        reward = 0.0
    details = {
        "backbone_id": backbone_id, "node_f1": node_f1, "state_f1": state_f1,
        "edge_f1": edge_f1, "cpd_kl": cpd_kl, "cpd_reward": cpd_reward,
        "comparable_cpd_columns": comparable_columns, "dag_valid": validation.dag_valid,
        "schema_valid": validation.schema_valid, "counts": counts,
        "predicted_edge_denominator": pred_edge_count,
        "excess_complexity_penalty": excess, "invalidity_penalty": invalidity,
        "final_reward": reward, "validation_errors": validation.errors,
    }
    if config.log_details:
        printable = dict(details)
        if math.isinf(printable["cpd_kl"]):
            printable["cpd_kl"] = "inf"
        print("[reward] " + json.dumps(printable, ensure_ascii=False, sort_keys=True))
    return reward, details


# Runtime state used by TRL's callback.
_REWARD_CONFIG = RewardConfig()
_ALIGNER = SemanticAligner()


def reward_fn(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    del prompts
    targets = kwargs.get("backbone_json", [])
    ids = kwargs.get("backbone_id", [])
    if not (len(completions) == len(targets) == len(ids)):
        raise ValueError(
            "reward metadata length mismatch: "
            f"completions={len(completions)} backbone_json={len(targets)} backbone_id={len(ids)}"
        )
    return [compute_reward(completion, json.loads(target), str(case_id), _ALIGNER, _REWARD_CONFIG)[0]
            for completion, target, case_id in zip(completions, targets, ids)]


# ---------------------------------------------------------------------------
# CLI and training
# ---------------------------------------------------------------------------

def validate_unit_sum(name: str, values: Mapping[str, float]) -> None:
    if any(value < 0 for value in values.values()):
        raise ValueError(f"{name} weights must be nonnegative: {dict(values)}")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} weights must sum to one, got {sum(values.values()):.6f}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a subgraph-set to backbone model with QLoRA GRPO")
    parser.add_argument("--prism-json", default=str(DEFAULT_PRISM_JSON))
    parser.add_argument("--backbone-jsonl", default=str(DEFAULT_BACKBONE_JSONL))
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations-per-case", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--skip-overlength", action="store_true")
    parser.add_argument("--w-node", type=float, default=.25)
    parser.add_argument("--w-state", type=float, default=.25)
    parser.add_argument("--w-edge", type=float, default=.25)
    parser.add_argument("--w-cpd", type=float, default=.25)
    parser.add_argument("--lambda-complexity", type=float, default=.05)
    parser.add_argument("--w-complexity-nodes", type=float, default=.15)
    parser.add_argument("--w-complexity-edges", type=float, default=.35)
    parser.add_argument("--w-complexity-states", type=float, default=.15)
    parser.add_argument("--w-complexity-parameters", type=float, default=.35)
    parser.add_argument("--invalid-penalty", type=float, default=1.0)
    parser.add_argument("--invalid-reward-mode", choices=("zero", "penalty"), default="zero")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--log-reward-details", action="store_true")
    parser.add_argument("--validate-dataset", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float]]:
    fidelity = {"nodes": args.w_node, "states": args.w_state, "edges": args.w_edge, "cpd": args.w_cpd}
    complexity = {
        "nodes": args.w_complexity_nodes, "edges": args.w_complexity_edges,
        "states": args.w_complexity_states, "parameters": args.w_complexity_parameters,
    }
    validate_unit_sum("fidelity", fidelity)
    validate_unit_sum("complexity", complexity)
    for name in ("max_steps", "grad_accum", "num_generations", "permutations_per_case",
                 "max_prompt_tokens", "max_completion_tokens", "judge_retries", "eval_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.lambda_complexity < 0 or args.invalid_penalty < 0:
        raise ValueError("learning rate must be positive and penalties nonnegative")
    return fidelity, complexity


def make_judge(model: str, retries: int) -> Callable[[str, str], Mapping[str, Any] | None]:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is required for semantic judging; use --exact-only to disable it")
    try:
        from huggingface_hub import InferenceClient
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for semantic judging") from error
    client = InferenceClient(api_key=token)

    def judge(system: str, prompt: str) -> Mapping[str, Any] | None:
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model, max_tokens=2000,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                )
                return extract_json_robust(response.choices[0].message.content)
            except Exception as error:  # bounded, explicit failure after final attempt
                if attempt + 1 == retries:
                    print(f"[judge warning] semantic judgment failed after {retries} attempts: {type(error).__name__}")
        return None
    return judge


def load_and_group(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_subgraphs(args.prism_json)
    backbones = load_backbones(args.backbone_jsonl)
    return group_subgraph_rows(rows, backbones)


def _print_report(report: Mapping[str, Any]) -> None:
    qualifier = "estimated " if report["estimated"] else ""
    print(
        f"Prompt lengths ({qualifier}tokens): min={report['minimum']} "
        f"median={report['median']} p95={report['p95']} max={report['maximum']}"
    )
    if report["over_limit_ids"]:
        print(f"Over-limit backbone IDs ({len(report['over_limit_ids'])}): " + ", ".join(report["over_limit_ids"]))


def validate_dataset_mode(args: argparse.Namespace, cases: Sequence[Mapping[str, Any]]) -> None:
    subgraph_counts = [len(case["subgraphs"]) for case in cases]
    errors = []
    missing_target_cpds = 0
    for case in cases:
        for index, graph in enumerate(case["subgraphs"], 1):
            result = validate_graph(graph, require_cpds=True)
            if not result.valid:
                errors.append(f"{case['backbone_id']} subgraph {index}: {result.errors}")
        result = validate_graph(case["backbone"], require_cpds=args.w_cpd > 0)
        if not result.valid:
            errors.append(f"{case['backbone_id']} backbone: {result.errors}")
        if len(case["backbone"].get("cpds", [])) < len(case["backbone"].get("edges", [])):
            missing_target_cpds += 1
    examples = augment_cases(cases, args.permutations_per_case, args.seed)
    leaks = [item["backbone_id"] for item in examples if item["backbone_json"] in item["prompt"]]
    report = prompt_length_report(examples, None, args.max_prompt_tokens)
    print(f"Backbone cases: {len(cases)}")
    print(
        f"Subgraphs per backbone: min={min(subgraph_counts)} median={statistics.median(subgraph_counts)} "
        f"max={max(subgraph_counts)} distribution={dict(sorted(Counter(subgraph_counts).items()))}"
    )
    _print_report(report)
    print(f"Target leakage checks: {'PASS' if not leaks else 'FAIL'}")
    try:
        splits = read_saved_splits(args.split_root)
    except RuntimeError as error:
        print(f"Split verification unavailable: {error}")
        splits = None
    if splits is not None:
        all_case_ids = set().union(*splits.values())
        expected_ids = {str(case["backbone_id"]) for case in cases}
        if all_case_ids != expected_ids:
            errors.append(f"saved splits cover {len(all_case_ids)} IDs, expected {len(expected_ids)}")
        print("Split leakage check: PASS (train/val/test are backbone-disjoint)")
    if missing_target_cpds and args.w_cpd > 0:
        print(f"WARNING: {missing_target_cpds} backbones lack one or more CPDs; use --w-cpd 0 and renormalize weights")
    if leaks:
        errors.append(f"serialized targets appeared in prompts for {leaks}")
    if report["over_limit_ids"] and not args.skip_overlength:
        errors.append(
            f"{len(report['over_limit_ids'])} cases exceed --max-prompt-tokens; "
            "increase it or pass --skip-overlength explicitly"
        )
    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:30])
        raise ValueError(f"dataset validation failed with {len(errors)} issue(s):\n{preview}")
    print("Dataset validation: PASS")


def run_training(
    args: argparse.Namespace, cases: Sequence[Mapping[str, Any]],
    fidelity_weights: Mapping[str, float], complexity_weights: Mapping[str, float],
) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.trainer_utils import get_last_checkpoint
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError(f"training dependency unavailable: {error}") from error

    global _ALIGNER, _REWARD_CONFIG
    _ALIGNER = SemanticAligner(None if args.exact_only else make_judge(args.judge_model, args.judge_retries))
    _REWARD_CONFIG = RewardConfig(
        node_weight=fidelity_weights["nodes"], state_weight=fidelity_weights["states"],
        edge_weight=fidelity_weights["edges"], cpd_weight=fidelity_weights["cpd"],
        lambda_complexity=args.lambda_complexity, complexity_weights=complexity_weights,
        invalid_penalty=args.invalid_penalty, invalid_reward_mode=args.invalid_reward_mode,
        require_cpds=args.w_cpd > 0, log_details=args.log_reward_details or args.smoke,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_ids = read_saved_splits(args.split_root)
    if split_ids is None:
        split_ids = deterministic_case_split([str(case["backbone_id"]) for case in cases], args.seed)
        check_split_leakage(split_ids)
        print("No saved split directories found; using a deterministic 80/10/10 backbone-level split")
    train_cases = [case for case in cases if case["backbone_id"] in split_ids["train"]]
    val_cases = [case for case in cases if case["backbone_id"] in split_ids["val"]]
    if not train_cases or not val_cases:
        raise ValueError(
            f"backbone-level split is empty: train={len(train_cases)} val={len(val_cases)}"
        )
    if args.smoke:
        train_cases = train_cases[:2]
        val_cases = val_cases[:1]
    train_examples = augment_cases(
        train_cases, args.permutations_per_case, args.seed, tokenizer
    )
    # Validation uses one stable order per case rather than augmented copies.
    val_examples = augment_cases(val_cases, 1, args.seed, tokenizer)
    train_examples, train_report = filter_overlength(
        train_examples, tokenizer, args.max_prompt_tokens, args.skip_overlength
    )
    val_examples, val_report = filter_overlength(
        val_examples, tokenizer, args.max_prompt_tokens, args.skip_overlength
    )
    print(f"Backbone split: train={len(train_cases)} val={len(val_cases)} "
          f"test={len(split_ids['test'])}")
    print("Training prompts:")
    _print_report(train_report)
    print("Validation prompts:")
    _print_report(val_report)
    if not train_examples or not val_examples:
        raise ValueError("no training examples remain after prompt-length filtering")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quantization, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        token=os.getenv("HF_TOKEN") or None,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    output_dir = args.output_dir
    max_steps, save_steps = args.max_steps, args.save_steps
    if args.smoke:
        output_dir = str(Path(args.output_dir).with_name(Path(args.output_dir).name + "_smoke"))
        max_steps, save_steps = min(args.max_steps, 4), 0
    config_kwargs = dict(
        output_dir=output_dir, max_steps=max_steps, num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_steps=50,
        num_generations=args.num_generations, max_prompt_length=args.max_prompt_tokens,
        max_completion_length=args.max_completion_tokens, temperature=.9, top_p=.95,
        bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
        logging_steps=1 if args.smoke else 5, save_strategy="steps" if save_steps else "no",
        eval_strategy="steps", eval_steps=1 if args.smoke else args.eval_steps,
        per_device_eval_batch_size=1,
        save_total_limit=3, remove_unused_columns=False,
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [], seed=args.seed,
    )
    if save_steps:
        config_kwargs["save_steps"] = save_steps
    config = GRPOConfig(**config_kwargs)
    trainer = GRPOTrainer(
        model=model, args=config, train_dataset=train_dataset, eval_dataset=val_dataset,
        reward_funcs=reward_fn, processing_class=tokenizer,
    )
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint:
        print(f"Resuming backbone training from {last_checkpoint}")
    else:
        print(f"Initializing backbone training from extraction adapter {args.adapter_path}")
    trainer.train(resume_from_checkpoint=last_checkpoint)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    fidelity, complexity = validate_args(args)
    cases = load_and_group(args)
    if args.validate_dataset:
        validate_dataset_mode(args, cases)
        return
    run_training(args, cases, fidelity, complexity)


if __name__ == "__main__":
    main()
