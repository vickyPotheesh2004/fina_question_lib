"""
question_lib/dependency_graph.py
Build a DAG of sub-formulas and topologically sort them.

For multi-step questions like:
  "Assuming gross margin grows at 2x revenue growth, what is FY24 margin?"

We need:
  step 1: revenue_growth_yoy   (depends on rev_2023, rev_2022)
  step 2: margin_growth_rate   (depends on step 1)
  step 3: gross_margin_2023    (depends on gp_2023, rev_2023)
  step 4: gross_margin_2024    (depends on step 2 + step 3)

Topological sort gives the execution order. Cycle detection guards
against logical errors.

Pure stdlib. NO LLM. < 1 ms for 20 sub-formulas.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .models import SubFormula


@dataclass
class GraphNode:
    name: str
    sub_formula: SubFormula
    deps: Set[str] = field(default_factory=set)        # incoming edges
    dependents: Set[str] = field(default_factory=set)  # outgoing edges


@dataclass
class Graph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)

    def add(self, sub: SubFormula):
        if sub.name in self.nodes:
            return
        node = GraphNode(
            name=sub.name,
            sub_formula=sub,
            deps=set(sub.depends_on or []),
        )
        self.nodes[sub.name] = node

    def link(self):
        """Populate `dependents` from `deps`."""
        for node in self.nodes.values():
            for d in node.deps:
                if d in self.nodes:
                    self.nodes[d].dependents.add(node.name)

    def topological_sort(self) -> Tuple[List[str], List[str]]:
        """Kahn's algorithm. Returns (order, cycle_nodes).

        If the graph has cycles, `cycle_nodes` lists the unresolvable nodes.
        """
        in_deg: Dict[str, int] = {n: 0 for n in self.nodes}
        for node in self.nodes.values():
            for d in node.deps:
                if d in self.nodes:
                    in_deg[node.name] += 1

        queue: deque = deque([n for n, d in in_deg.items() if d == 0])
        order: List[str] = []

        while queue:
            n = queue.popleft()
            order.append(n)
            for m in self.nodes[n].dependents:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)

        cycle_nodes = [n for n, d in in_deg.items() if d > 0]
        return order, cycle_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(sub_formulas: List[SubFormula]) -> Graph:
    g = Graph()
    for sf in sub_formulas:
        g.add(sf)
    g.link()
    return g


def execution_order(sub_formulas: List[SubFormula]) -> Tuple[List[SubFormula], List[str]]:
    """Return (sorted_subs, cycle_nodes)."""
    g = build_graph(sub_formulas)
    order, cycles = g.topological_sort()
    by_name = {s.name: s for s in sub_formulas}
    sorted_subs = [by_name[n] for n in order if n in by_name]
    return sorted_subs, cycles


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .models import Operation

    subs = [
        SubFormula(name="gross_margin_2024",
                   formula_id="margin_projected",
                   depends_on=["margin_growth_rate", "gross_margin_2023"]),
        SubFormula(name="revenue_growth_yoy",
                   formula_id="growth_yoy",
                   inputs=["rev_2023", "rev_2022"]),
        SubFormula(name="margin_growth_rate",
                   formula_id="multiply",
                   depends_on=["revenue_growth_yoy"]),
        SubFormula(name="gross_margin_2023",
                   formula_id="gross_margin",
                   inputs=["gp_2023", "rev_2023"]),
    ]

    print("dependency_graph — self test")
    sorted_subs, cycles = execution_order(subs)
    for i, s in enumerate(sorted_subs, 1):
        print(f"  {i}. {s.name:<22} deps={list(s.depends_on)}")
    if cycles:
        print(f"  CYCLES: {cycles}")
    else:
        print("  no cycles, sorted OK")
