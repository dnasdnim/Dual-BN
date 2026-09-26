"""Entry point and metrics for prompted maximal-backbone evaluation.

When executed, this file runs the shared GPT-6 Astra/Qwen few-shot experiment
implemented in ``evaluate_maximal_subgraph_union_prompted.py``. The deterministic
union helpers below remain importable only as target-blind metric utilities.


The construction phase receives only submitted subgraphs.  It merges labels by
the repository's deterministic normalization, unions nodes, states, and directed
edges, and combines non-conflicting CPD cells.  Ground-truth backbones are used
only after construction, during evaluation.

This is an evidence-coverage experiment, not a latent-graph inference method:
missing backbone content remains missing rather than being guessed.
"""
from __future__ import annotations

if __name__ == "__main__":
    from evaluate_maximal_subgraph_union_prompted import main as prompted_main
    prompted_main()
    raise SystemExit

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import train_backbone_grpo as core


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRISM_JSON = SCRIPT_DIR / "prism_bn_openai.json"
DEFAULT_BACKBONE_JSONL = SCRIPT_DIR / "bn_marginal_raw_openai.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_outputs" / "maximal_union_openai"


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalized_unique(values: Iterable[str], what: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        key = core.normalized_name(value)
        if not key:
            raise ValueError(f"{what} has an empty normalized label: {value!r}")
        previous = result.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                f"ambiguous {what} labels normalize identically: {previous!r}, {value!r}"
            )
        result[key] = value
    return result


def _node_index(graph: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, Mapping) or not isinstance(node.get("node"), str):
            raise ValueError("every node must have a string 'node' label")
        key = core.normalized_name(node["node"])
        if not key or key in result:
            raise ValueError(f"duplicate or empty normalized node label: {node['node']!r}")
        states = node.get("states")
        if not isinstance(states, list) or not states or not all(isinstance(x, str) for x in states):
            raise ValueError(f"node {node['node']!r} requires nonempty string states")
        _normalized_unique(states, f"states for node {node['node']!r}")
        result[key] = node
    return result


def _metric(correct: int, predicted: int, target: int) -> Dict[str, float | int]:
    precision = correct / predicted if predicted else (1.0 if target == 0 else 0.0)
    recall = correct / target if target else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correct": correct,
        "predicted": predicted,
        "target": target,
        "missed": target - correct,
        "extra": predicted - correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_maximal_graph(
    subgraphs: Sequence[Mapping[str, Any]], cpd_tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Union submitted graphs without consulting or inferring the backbone."""
    if not subgraphs:
        raise ValueError("cannot construct a maximal graph from zero subgraphs")
    if cpd_tolerance < 0:
        raise ValueError("cpd_tolerance must be nonnegative")

    # Sorting by content makes representative labels and CPD values independent
    # of source-file ordering.
    ordered = sorted((core.canonical_graph(graph) for graph in subgraphs), key=core.stable_json)
    names: Dict[str, str] = {}
    states: Dict[str, Dict[str, str]] = {}
    graph_node_indexes: List[Dict[str, Mapping[str, Any]]] = []

    for graph in ordered:
        local = _node_index(graph)
        graph_node_indexes.append(local)
        for node_key, node in local.items():
            names.setdefault(node_key, str(node["node"]))
            state_bucket = states.setdefault(node_key, {})
            for state in node["states"]:
                state_bucket.setdefault(core.normalized_name(state), state)

    edges: set[Tuple[str, str]] = set()
    # Each cell retains all numeric observations so conflicts cannot be hidden.
    cpd_cells: Dict[Tuple[str, str], Dict[Tuple[str, str], List[float]]] = {}
    cpd_null_cells: Dict[Tuple[str, str], set[Tuple[str, str]]] = {}
    cpd_observations: Dict[Tuple[str, str], int] = {}

    for graph, local in zip(ordered, graph_node_indexes):
        for edge in graph["edges"]:
            parent = core.normalized_name(str(edge.get("parent", "")))
            child = core.normalized_name(str(edge.get("child", "")))
            if parent not in local or child not in local:
                raise ValueError(f"edge {parent!r}->{child!r} references a missing local node")
            edges.add((parent, child))

        for cpd in graph["cpds"]:
            parent = core.normalized_name(str(cpd.get("parent", "")))
            child = core.normalized_name(str(cpd.get("child", "")))
            edge = (parent, child)
            if edge not in edges or parent not in local or child not in local:
                raise ValueError(f"CPD {parent!r}->{child!r} has no valid local edge")
            matrix = cpd.get("matrix")
            parent_states = list(local[parent]["states"])
            child_states = list(local[child]["states"])
            rows, columns = core._matrix_shape(matrix)
            if (rows, columns) != (len(child_states), len(parent_states)):
                raise ValueError(
                    f"CPD {parent!r}->{child!r} shape {(rows, columns)} != "
                    f"{(len(child_states), len(parent_states))}"
                )
            cpd_observations[edge] = cpd_observations.get(edge, 0) + 1
            values = cpd_cells.setdefault(edge, {})
            nulls = cpd_null_cells.setdefault(edge, set())
            for row, child_state in enumerate(child_states):
                for column, parent_state in enumerate(parent_states):
                    cell = (core.normalized_name(child_state), core.normalized_name(parent_state))
                    value = matrix[row][column]
                    if value is None:
                        nulls.add(cell)
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"CPD {parent!r}->{child!r} has nonnumeric value {value!r}")
                    numeric = float(value)
                    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                        raise ValueError(f"CPD {parent!r}->{child!r} has invalid probability {value!r}")
                    values.setdefault(cell, []).append(numeric)

    conflicts = []
    cpds = []
    for edge in sorted(cpd_cells):
        parent, child = edge
        values = cpd_cells[edge]
        for cell, observations in values.items():
            spread = max(observations) - min(observations)
            if spread > cpd_tolerance:
                conflicts.append({
                    "parent": names[parent], "child": names[child],
                    "child_state": states[child][cell[0]],
                    "parent_state": states[parent][cell[1]],
                    "minimum": min(observations), "maximum": max(observations),
                    "spread": spread,
                })
        matrix = []
        observed = 0
        for child_state in states[child]:
            row = []
            for parent_state in states[parent]:
                observations = values.get((child_state, parent_state), [])
                if observations:
                    # Values within tolerance are equivalent evidence. Retain a
                    # deterministic observed value rather than fabricating precision.
                    row.append(sorted(observations)[0])
                    observed += 1
                else:
                    row.append(None)
            matrix.append(row)
        total = len(states[child]) * len(states[parent])
        cpds.append({
            "parent": names[parent], "child": names[child], "matrix": matrix,
            "observed_cells": observed, "total_cells": total,
            "complete": observed == total,
            "source_occurrences": cpd_observations[edge],
        })

    if conflicts:
        preview = conflicts[:5]
        raise ValueError(
            f"found {len(conflicts)} conflicting CPD cells beyond tolerance "
            f"{cpd_tolerance}: {preview}"
        )

    nodes = [
        {"node": names[key], "states": list(states[key].values())}
        for key in sorted(names)
    ]
    output_edges = [
        {"parent": names[parent], "child": names[child]}
        for parent, child in sorted(edges)
    ]
    represented_cells = sum(cpd["observed_cells"] for cpd in cpds)
    possible_cells = sum(cpd["total_cells"] for cpd in cpds)
    return {
        "construction": "deterministic_normalized_union",
        "nodes": nodes,
        "edges": output_edges,
        "cpds": cpds,
        "evidence_summary": {
            "input_subgraphs": len(subgraphs),
            "cpd_edges": len(cpds),
            "observed_cpd_cells": represented_cells,
            "possible_cpd_cells": possible_cells,
            "cpd_conflicts": 0,
        },
    }


def _edge_index(graph: Mapping[str, Any]) -> set[Tuple[str, str]]:
    return {
        (core.normalized_name(str(edge["parent"])), core.normalized_name(str(edge["child"])))
        for edge in graph.get("edges", [])
    }


def _cpd_metrics(
    predicted: Mapping[str, Any], target: Mapping[str, Any],
) -> Dict[str, Any]:
    pred_nodes = _node_index(predicted)
    target_nodes = _node_index(target)
    pred_cpds = {
        (core.normalized_name(str(item["parent"])), core.normalized_name(str(item["child"]))): item
        for item in predicted.get("cpds", [])
    }
    target_cpds = {
        (core.normalized_name(str(item["parent"])), core.normalized_name(str(item["child"]))): item
        for item in target.get("cpds", [])
    }
    absolute_errors: List[float] = []
    kls: List[float] = []
    target_cells = 0
    comparable_cells = 0
    complete_columns = 0
    complete_edges = 0

    for edge, target_cpd in target_cpds.items():
        target_parent, target_child = target_nodes[edge[0]], target_nodes[edge[1]]
        target_matrix = target_cpd["matrix"]
        target_parent_states = list(target_parent["states"])
        target_child_states = list(target_child["states"])
        target_cells += sum(
            value is not None for row in target_matrix for value in row
        )
        pred_cpd = pred_cpds.get(edge)
        if pred_cpd is None or edge[0] not in pred_nodes or edge[1] not in pred_nodes:
            continue
        pred_parent_states = {
            core.normalized_name(state): index
            for index, state in enumerate(pred_nodes[edge[0]]["states"])
        }
        pred_child_states = {
            core.normalized_name(state): index
            for index, state in enumerate(pred_nodes[edge[1]]["states"])
        }
        pred_matrix = pred_cpd["matrix"]
        edge_target_cells = 0
        edge_comparable_cells = 0
        for target_column, parent_state in enumerate(target_parent_states):
            pcol = pred_parent_states.get(core.normalized_name(parent_state))
            actual_column = []
            predicted_column = []
            column_complete = pcol is not None
            for target_row, child_state in enumerate(target_child_states):
                actual = target_matrix[target_row][target_column]
                if actual is None:
                    continue
                edge_target_cells += 1
                prow = pred_child_states.get(core.normalized_name(child_state))
                value = None if pcol is None or prow is None else pred_matrix[prow][pcol]
                if value is None:
                    column_complete = False
                    continue
                numeric_actual, numeric_predicted = float(actual), float(value)
                absolute_errors.append(abs(numeric_actual - numeric_predicted))
                comparable_cells += 1
                edge_comparable_cells += 1
                actual_column.append(numeric_actual)
                predicted_column.append(numeric_predicted)
            if column_complete and actual_column and len(actual_column) == len(target_child_states):
                complete_columns += 1
                kls.append(core.kl_divergence(actual_column, predicted_column))
        if edge_target_cells and edge_comparable_cells == edge_target_cells:
            complete_edges += 1

    matched_cpd_edges = set(pred_cpds) & set(target_cpds)
    return {
        "predicted_edges": len(pred_cpds),
        "target_edges": len(target_cpds),
        "matched_edges": len(matched_cpd_edges),
        "missing_edges": len(set(target_cpds) - set(pred_cpds)),
        "extra_edges": len(set(pred_cpds) - set(target_cpds)),
        "complete_target_edges": complete_edges,
        "target_numeric_cells": target_cells,
        "comparable_numeric_cells": comparable_cells,
        "missing_numeric_cells": target_cells - comparable_cells,
        "cell_coverage": comparable_cells / target_cells if target_cells else 1.0,
        "mean_absolute_error": statistics.fmean(absolute_errors) if absolute_errors else None,
        "max_absolute_error": max(absolute_errors) if absolute_errors else None,
        "complete_columns": complete_columns,
        "mean_column_kl": statistics.fmean(kls) if kls else None,
    }


def evaluate_maximal_graph(
    predicted: Mapping[str, Any], target: Mapping[str, Any],
) -> Dict[str, Any]:
    pred_nodes = _node_index(predicted)
    target_nodes = _node_index(target)
    pred_node_keys, target_node_keys = set(pred_nodes), set(target_nodes)
    matched_nodes = pred_node_keys & target_node_keys

    missing_nodes = [target_nodes[key]["node"] for key in sorted(target_node_keys - pred_node_keys)]
    extra_nodes = [pred_nodes[key]["node"] for key in sorted(pred_node_keys - target_node_keys)]
    node_metrics = {
        **_metric(len(matched_nodes), len(pred_nodes), len(target_nodes)),
        "missing_labels": missing_nodes,
        "extra_labels": extra_nodes,
    }

    predicted_state_count = sum(len(node["states"]) for node in pred_nodes.values())
    target_state_count = sum(len(node["states"]) for node in target_nodes.values())
    matched_state_count = 0
    missing_states: Dict[str, List[str]] = {}
    extra_states: Dict[str, List[str]] = {}
    for key in sorted(target_node_keys | pred_node_keys):
        predicted_states = (
            _normalized_unique(pred_nodes[key]["states"], "predicted states")
            if key in pred_nodes else {}
        )
        target_states = (
            _normalized_unique(target_nodes[key]["states"], "target states")
            if key in target_nodes else {}
        )
        matched_state_count += len(set(predicted_states) & set(target_states))
        missing = [target_states[x] for x in sorted(set(target_states) - set(predicted_states))]
        extra = [predicted_states[x] for x in sorted(set(predicted_states) - set(target_states))]
        if missing:
            missing_states[str(target_nodes[key]["node"])] = missing
        if extra:
            extra_states[str(pred_nodes[key]["node"])] = extra
    state_metrics = {
        **_metric(matched_state_count, predicted_state_count, target_state_count),
        "missing_by_node": missing_states,
        "extra_by_node": extra_states,
    }

    predicted_edges, target_edges = _edge_index(predicted), _edge_index(target)
    matched_edges = predicted_edges & target_edges
    missing_edges = target_edges - predicted_edges
    extra_edges = predicted_edges - target_edges

    def edge_record(edge: Tuple[str, str], index: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
        return {"parent": str(index[edge[0]]["node"]), "child": str(index[edge[1]]["node"])}

    missing_due_to_nodes = []
    missing_relationships = []
    for edge in sorted(missing_edges):
        record = edge_record(edge, target_nodes)
        absent = [target_nodes[key]["node"] for key in edge if key not in pred_nodes]
        if absent:
            record["absent_endpoints"] = list(dict.fromkeys(map(str, absent)))
            missing_due_to_nodes.append(record)
        else:
            missing_relationships.append(record)
    edge_metrics = {
        **_metric(len(matched_edges), len(predicted_edges), len(target_edges)),
        "missing_due_to_absent_nodes": missing_due_to_nodes,
        "missing_relationships_between_present_nodes": missing_relationships,
        "extra_edges": [edge_record(edge, pred_nodes) for edge in sorted(extra_edges)],
    }

    if predicted.get("skip_cpd_evaluation"):
        cpd_metrics: Dict[str, Any] = {
            "skipped": True,
            "reason": str(predicted.get(
                "cpd_skip_reason", "prediction contains a non-probability"
            )),
        }
    else:
        cpd_metrics = {"skipped": False, **_cpd_metrics(predicted, target)}

    return {
        "nodes": node_metrics,
        "states": state_metrics,
        "edges": edge_metrics,
        "cpds": cpd_metrics,
        "full_coverage": {
            "nodes": node_metrics["missed"] == 0,
            "states": state_metrics["missed"] == 0,
            "edges": edge_metrics["missed"] == 0,
            "nodes_states_edges": all(
                metrics["missed"] == 0
                for metrics in (node_metrics, state_metrics, edge_metrics)
            ),
        },
    }


def aggregate_metrics(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {"case_count": len(cases)}
    for component in ("nodes", "states", "edges"):
        rows = [case["metrics"][component] for case in cases]
        correct = sum(int(row["correct"]) for row in rows)
        predicted = sum(int(row["predicted"]) for row in rows)
        target = sum(int(row["target"]) for row in rows)
        aggregate[component] = {
            **_metric(correct, predicted, target),
            "macro_precision": statistics.fmean(float(row["precision"]) for row in rows),
            "macro_recall": statistics.fmean(float(row["recall"]) for row in rows),
            "macro_f1": statistics.fmean(float(row["f1"]) for row in rows),
            "full_coverage_cases": sum(bool(case["metrics"]["full_coverage"][component]) for case in cases),
        }
    all_cpd_rows = [case["metrics"]["cpds"] for case in cases]
    cpd_rows = [row for row in all_cpd_rows if not row.get("skipped", False)]
    target_cells = sum(int(row["target_numeric_cells"]) for row in cpd_rows)
    comparable_cells = sum(int(row["comparable_numeric_cells"]) for row in cpd_rows)
    aggregate["cpds"] = {
        "evaluated_cases": len(cpd_rows),
        "skipped_cases": len(all_cpd_rows) - len(cpd_rows),
        "predicted_edges": sum(int(row["predicted_edges"]) for row in cpd_rows),
        "target_edges": sum(int(row["target_edges"]) for row in cpd_rows),
        "matched_edges": sum(int(row["matched_edges"]) for row in cpd_rows),
        "complete_target_edges": sum(int(row["complete_target_edges"]) for row in cpd_rows),
        "target_numeric_cells": target_cells,
        "comparable_numeric_cells": comparable_cells,
        "missing_numeric_cells": target_cells - comparable_cells,
        "cell_coverage": (
            comparable_cells / target_cells if target_cells
            else (1.0 if cpd_rows else None)
        ),
    }
    aggregate["full_coverage_cases"] = sum(
        bool(case["metrics"]["full_coverage"]["nodes_states_edges"])
        for case in cases
    )
    return aggregate


def _csv_row(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = result["metrics"]
    return {
        "backbone_id": result["backbone_id"],
        "source_id": result["source_id"],
        "subgraphs": result["subgraph_count"],
        "node_predicted": metrics["nodes"]["predicted"],
        "node_target": metrics["nodes"]["target"],
        "node_missed": metrics["nodes"]["missed"],
        "node_extra": metrics["nodes"]["extra"],
        "node_recall": metrics["nodes"]["recall"],
        "state_predicted": metrics["states"]["predicted"],
        "state_target": metrics["states"]["target"],
        "state_missed": metrics["states"]["missed"],
        "state_extra": metrics["states"]["extra"],
        "state_recall": metrics["states"]["recall"],
        "edge_predicted": metrics["edges"]["predicted"],
        "edge_target": metrics["edges"]["target"],
        "edge_missed": metrics["edges"]["missed"],
        "edge_extra": metrics["edges"]["extra"],
        "edge_recall": metrics["edges"]["recall"],
        "edge_missed_absent_node": len(metrics["edges"]["missing_due_to_absent_nodes"]),
        "edge_missed_present_nodes": len(metrics["edges"]["missing_relationships_between_present_nodes"]),
        "cpd_skipped": metrics["cpds"].get("skipped", False),
        "cpd_cell_coverage": metrics["cpds"].get("cell_coverage"),
        "complete": metrics["full_coverage"]["nodes_states_edges"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prism-json", default=str(DEFAULT_PRISM_JSON))
    parser.add_argument("--backbone-jsonl", default=str(DEFAULT_BACKBONE_JSONL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case-id", action="append", default=[], help="Only evaluate this backbone ID; repeatable")
    parser.add_argument("--exclude-case-id", action="append", default=[], help="Exclude a demonstration/case ID; repeatable")
    parser.add_argument("--cpd-tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.cpd_tolerance < 0:
        parser.error("--cpd-tolerance must be nonnegative")
    overlap = set(args.case_id) & set(args.exclude_case_id)
    if overlap:
        parser.error(f"case IDs cannot be both included and excluded: {sorted(overlap)}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    # Join validation occurs here, but case['backbone'] is deliberately not
    # passed to build_maximal_graph below.
    cases = core.group_subgraph_rows(
        core.load_subgraphs(args.prism_json), core.load_backbones(args.backbone_jsonl)
    )
    include, exclude = set(args.case_id), set(args.exclude_case_id)
    if include:
        found = {str(case["backbone_id"]) for case in cases}
        missing = include - found
        if missing:
            raise ValueError(f"requested case IDs are absent: {sorted(missing)}")
        cases = [case for case in cases if str(case["backbone_id"]) in include]
    cases = [case for case in cases if str(case["backbone_id"]) not in exclude]
    if not cases:
        raise ValueError("no cases remain after include/exclude filtering")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "run_config.json", {
        **vars(args),
        "selected_case_count": len(cases),
        "construction_uses_backbone": False,
    })

    results = []
    for case in cases:
        case_id = str(case["backbone_id"])
        source_id = str(case["source_id"])
        maximal = build_maximal_graph(case["subgraphs"], args.cpd_tolerance)
        # First and only target access for this case's computation.
        metrics = evaluate_maximal_graph(maximal, case["backbone"])
        result = {
            "backbone_id": case_id,
            "source_id": source_id,
            "subgraph_count": len(case["subgraphs"]),
            "metrics": metrics,
        }
        save_json(output / "graphs" / f"{case_id}.json", maximal)
        save_json(output / "cases" / f"{case_id}.json", result)
        results.append(result)
        print(
            f"{case_id}: nodes {metrics['nodes']['correct']}/{metrics['nodes']['target']}, "
            f"states {metrics['states']['correct']}/{metrics['states']['target']}, "
            f"edges {metrics['edges']['correct']}/{metrics['edges']['target']}",
            flush=True,
        )

    aggregate = aggregate_metrics(results)
    save_json(output / "summary.json", {"aggregate": aggregate, "cases": results})
    csv_rows = [_csv_row(result) for result in results]
    with open(output / "per_case_metrics.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps(aggregate, indent=2, allow_nan=False))
    print(f"Outputs saved to {output.resolve()}")


if __name__ == "__main__":
    main()
