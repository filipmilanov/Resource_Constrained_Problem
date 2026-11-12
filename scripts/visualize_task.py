import argparse
import ast
import html
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path

import webview
from pyvis.network import Network

from rckst.core.models import ConfigValue, RCKSTInstance, RCKSTSolution, RCKSTTask, RCKSTTaskConfig, RCKSTTaskOptions, TaskParams
from rckst.core.rckst_file_manager import RCKSTFileManager
from rckst.tasks import DeterministicConstructionTask

TASKS: dict[str, type[RCKSTTask]] = {
    "deterministic_construction": DeterministicConstructionTask,
}


def _parse_value(value: str) -> ConfigValue:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    return parsed if isinstance(parsed, str | int | float | bool) else value


def _parse_task_params(arguments: Sequence[str], parser: argparse.ArgumentParser) -> TaskParams:
    if len(arguments) % 2 != 0:
        parser.error("Task parameters must be provided as '--name value' pairs.")

    params: TaskParams = {}
    for key, value in zip(arguments[::2], arguments[1::2], strict=True):
        if not key.startswith("--") or len(key) == 2:
            parser.error(f"Invalid task parameter '{key}'. Expected '--name value'.")
        params[key[2:]] = _parse_value(value)
    return params


def _solution_edges(solution: RCKSTSolution) -> set[tuple[int, int]]:
    if solution.edges.ndim != 2 or solution.edges.shape[0] != 2:
        raise ValueError(f"Expected solution edges with shape [2, edge_count], got {tuple(solution.edges.shape)}.")
    return {tuple(sorted((int(source), int(target)))) for source, target in solution.edges.t().tolist()}


def _resource_row(instance: RCKSTInstance, node_id: int, one_based: bool) -> list[int]:
    row_index = node_id - 1 if one_based else node_id
    return [int(value) for value in instance.node_resources[row_index].tolist()]


def _evaluate_feasibility(instance: RCKSTInstance, solution: RCKSTSolution, one_based: bool) -> tuple[bool, list[int], list[int], list[int]]:
    raw_edges = [(int(source), int(target)) for source, target in solution.edges.t().tolist()]
    selected_edges = [tuple(sorted(edge)) for edge in raw_edges]
    selected_nodes = {node for edge in selected_edges for node in edge}
    instance_edges = {
        tuple(sorted((int(instance.edge_index[0, edge_idx]), int(instance.edge_index[1, edge_idx])))): edge_idx for edge_idx in range(instance.E)
    }
    valid_nodes = set(range(1, instance.n + 1) if one_based else range(instance.n))

    adjacency: dict[int, set[int]] = {node: set() for node in selected_nodes}
    for source, target in selected_edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited: set[int] = set()
    pending = [next(iter(selected_nodes))] if selected_nodes else []
    while pending:
        node = pending.pop()
        if node not in visited:
            visited.add(node)
            pending.extend(adjacency[node] - visited)

    structural_ok = (
        len(raw_edges) == instance.k - 1
        and len(set(selected_edges)) == len(selected_edges)
        and len(selected_nodes) == instance.k
        and selected_nodes <= valid_nodes
        and all(edge in instance_edges for edge in selected_edges)
        and visited == selected_nodes
    )

    yields = [0] * instance.m
    for node in selected_nodes & valid_nodes:
        resources = _resource_row(instance, node, one_based)
        yields = [total + value for total, value in zip(yields, resources, strict=True)]

    consumption = [0] * instance.m
    for edge in selected_edges:
        edge_idx = instance_edges.get(edge)
        if edge_idx is not None:
            costs = [int(value) for value in instance.edge_costs[edge_idx].tolist()]
            consumption = [total + value for total, value in zip(consumption, costs, strict=True)]

    slack = [available - used for available, used in zip(yields, consumption, strict=True)]
    return structural_ok and all(value >= 0 for value in slack), yields, consumption, slack


def build_graph_html(instance: RCKSTInstance, solution: RCKSTSolution, task_name: str, elapsed: float) -> str:
    selected_edges = _solution_edges(solution)
    selected_nodes = {node for edge in selected_edges for node in edge}

    edge_endpoints = instance.edge_index.flatten().tolist()
    one_based = bool(edge_endpoints) and min(edge_endpoints) >= 1 and max(edge_endpoints) <= instance.n
    node_ids = range(1, instance.n + 1) if one_based else range(instance.n)
    is_feasible, resource_yields, resource_consumption, resource_slack = _evaluate_feasibility(instance, solution, one_based)

    graph = Network(height="100vh", width="100%", bgcolor="#0b1020", font_color="#e5e7eb", cdn_resources="in_line", directed=False)
    graph.set_options(
        """
        {
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},
          "nodes": {"shape": "dot", "borderWidth": 2, "font": {"face": "Inter, Segoe UI, sans-serif", "size": 14}},
          "edges": {"smooth": {"enabled": true, "type": "continuous"}},
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -55, "centralGravity": 0.015, "springLength": 115, "springConstant": 0.08},
            "stabilization": {"enabled": true, "iterations": 700, "fit": true}
          }
        }
        """
    )

    for node_id in node_ids:
        is_selected = node_id in selected_nodes
        resources = _resource_row(instance, node_id, one_based)
        graph.add_node(
            node_id,
            label=f"{node_id}\nr={resources}",
            title=f"<b>Node {node_id}</b><br>Resources: {html.escape(str(resources))}",
            color={
                "background": "#ef4444" if is_selected else "#334155",
                "border": "#fecaca" if is_selected else "#64748b",
                "highlight": {"background": "#f87171", "border": "#ffffff"},
            },
            size=19 if is_selected else 13,
        )

    for edge_idx in range(instance.E):
        source = int(instance.edge_index[0, edge_idx])
        target = int(instance.edge_index[1, edge_idx])
        is_selected = tuple(sorted((source, target))) in selected_edges
        weight = int(instance.edge_weights[edge_idx])
        costs = [int(value) for value in instance.edge_costs[edge_idx].tolist()]
        graph.add_edge(
            source,
            target,
            label=f"w={weight} | c={costs}",
            title=f"<b>{source} — {target}</b><br>Weight: {weight}<br>Resource costs: {html.escape(str(costs))}",
            color={"color": "#ef4444" if is_selected else "#475569", "highlight": "#fca5a5" if is_selected else "#94a3b8"},
            font={"color": "#cbd5e1", "size": 10, "strokeWidth": 3, "strokeColor": "#0b1020", "align": "top"},
            width=4 if is_selected else 1.2,
        )

    generated_html = graph.generate_html(notebook=False)
    generated_html = re.sub(r'<link\s+[^>]*bootstrap[^>]*/>', "", generated_html, flags=re.DOTALL)
    generated_html = re.sub(r'<script\s+[^>]*bootstrap[^>]*>.*?</script>', "", generated_html, flags=re.DOTALL)
    summary = f"""
    <aside class="rckst-summary">
      <div class="rckst-eyebrow">RCKST VISUALIZER</div>
      <h1>{html.escape(instance.name)}</h1>
      <div class="rckst-task">{html.escape(task_name)}</div>
      <dl>
        <div><dt>Nodes</dt><dd>{instance.n}</dd></div>
        <div><dt>Edges</dt><dd>{instance.E}</dd></div>
        <div><dt>Required k</dt><dd>{instance.k}</dd></div>
        <div><dt>Selected</dt><dd>{len(selected_nodes)}</dd></div>
        <div><dt>Weight</dt><dd>{html.escape(str(solution.total_weight))}</dd></div>
        <div><dt>Runtime</dt><dd>{elapsed:.3f}s</dd></div>
        <div><dt>Feasible</dt><dd class="{'rckst-ok' if is_feasible else 'rckst-bad'}">{'Yes' if is_feasible else 'No'}</dd></div>
      </dl>
      <div class="rckst-resources">Yield {resource_yields}<br>Cost {resource_consumption}<br>Slack {resource_slack}</div>
      <div class="rckst-legend"><span></span> selected solution</div>
    </aside>
    """
    styles = """
    <style>
      html, body { margin: 0; overflow: hidden; background: #0b1020 !important; }
      #mynetwork { border: 0 !important; }
      .card { border: 0 !important; background: transparent !important; }
      .rckst-summary {
        position: fixed; z-index: 1000; top: 20px; left: 20px; width: 220px; padding: 18px;
        color: #e5e7eb; background: rgba(15, 23, 42, .88); border: 1px solid rgba(148, 163, 184, .2);
        border-radius: 16px; box-shadow: 0 18px 45px rgba(0, 0, 0, .35); backdrop-filter: blur(14px);
        font-family: Inter, "Segoe UI", sans-serif; pointer-events: none;
      }
      .rckst-eyebrow { color: #94a3b8; font-size: 10px; font-weight: 700; letter-spacing: .16em; }
      .rckst-summary h1 { margin: 7px 0 2px; font-size: 20px; line-height: 1.2; }
      .rckst-task { color: #94a3b8; font-size: 12px; }
      .rckst-summary dl { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 18px 0; }
      .rckst-summary dl div { display: flex; flex-direction: column; }
      .rckst-summary dt { color: #64748b; font-size: 10px; text-transform: uppercase; }
      .rckst-summary dd { margin: 2px 0 0; font-size: 14px; font-weight: 650; }
      .rckst-ok { color: #4ade80; }
      .rckst-bad { color: #f87171; }
      .rckst-resources { margin: 0 0 14px; padding: 10px; color: #cbd5e1; background: rgba(30, 41, 59, .8); border-radius: 9px; font-size: 11px; line-height: 1.6; }
      .rckst-legend { color: #cbd5e1; font-size: 11px; }
      .rckst-legend span { display: inline-block; width: 9px; height: 9px; margin-right: 7px; border-radius: 50%; background: #ef4444; }
    </style>
    """
    return generated_html.replace("</head>", f"{styles}</head>").replace("</body>", f"{summary}</body>")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one RCKST task and display its solution interactively.")
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("instance", type=Path)
    parser.add_argument("--seed", "-s", type=int, default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = _build_parser()
    args, task_arguments = parser.parse_known_args()
    params = _parse_task_params(task_arguments, parser)

    instance = RCKSTFileManager.load_instance(args.instance)
    task = TASKS[args.task]()
    config = RCKSTTaskConfig(instance=instance, options=RCKSTTaskOptions(seed=args.seed, params=params))

    logging.info("Running '%s' on '%s'", args.task, instance.name)
    started = time.perf_counter()
    solution = task.solve(config)
    elapsed = time.perf_counter() - started
    logging.info("Task completed in %.3fs", elapsed)

    graph_html = build_graph_html(instance, solution, args.task, elapsed)
    webview.create_window(
        f"RCKST — {instance.name}",
        html=graph_html,
        width=1280,
        height=800,
        min_size=(800, 600),
        background_color="#0b1020",
    )
    webview.start()


if __name__ == "__main__":
    main()
