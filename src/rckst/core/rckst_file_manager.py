from pathlib import Path

import torch
from jaxtyping import Int

from rckst.core.models import RCKSTInstance, RCKSTSolution


class RCKSTFileManager:
    @staticmethod
    def load_instance(path: Path) -> RCKSTInstance:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line]

        header_parts = lines[0].split()
        n = int(header_parts[0])
        E = int(header_parts[1])
        k = int(header_parts[2])
        m = int(header_parts[3])

        current_idx = 1
        if lines[current_idx] == "#node resources":
            current_idx += 1

        node_res_data: list[list[int]] = []
        for _ in range(n):
            node_res_data.append([int(x) for x in lines[current_idx].split()])
            current_idx += 1

        if current_idx < len(lines) and lines[current_idx] == "#edges":
            current_idx += 1

        edges_data: list[list[int]] = []
        weights_data: list[int] = []
        edge_costs_data: list[list[int]] = []

        for _ in range(E):
            parts = [int(x) for x in lines[current_idx].split()]
            edges_data.append([parts[0], parts[1]])
            weights_data.append(parts[2])
            edge_costs_data.append(parts[3:])
            current_idx += 1

        node_resources: Int[torch.Tensor, "n m"] = torch.tensor(node_res_data, dtype=torch.int64)
        edge_index: Int[torch.Tensor, "2 E"] = torch.tensor(edges_data, dtype=torch.int64).t().contiguous()
        edge_weights: Int[torch.Tensor, "E"] = torch.tensor(weights_data, dtype=torch.int64)
        edge_costs: Int[torch.Tensor, "E m"] = torch.tensor(edge_costs_data, dtype=torch.int64)

        return RCKSTInstance(name=path.stem, n=n, E=E, k=k, m=m, node_resources=node_resources, edge_index=edge_index, edge_weights=edge_weights, edge_costs=edge_costs)

    @staticmethod
    def save_solution(solution: RCKSTSolution, path: Path) -> None:
        lines = [solution.instance_name]
        edges_tensor = solution.edges.t()
        for row in edges_tensor:
            u, v = int(row[0].item()), int(row[1].item())
            lines.append(f"{u} {v}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
