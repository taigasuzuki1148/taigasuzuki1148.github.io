from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Optional

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister


@dataclass(frozen=True)
class ColoringLayout:
    """
    Helper for addressing one-hot variable qubits.

    Variables:
      x[v,c] = 1  <=>  vertex v has color c

    Qubit ordering in variable register:
      x[0,0], x[0,1], ..., x[0,k-1], x[1,0], ..., x[n-1,k-1]
    """
    n: int
    k: int
    x: QuantumRegister

    def q(self, v: int, c: int):
        """Return qubit for variable x[v,c]."""
        if not (0 <= v < self.n):
            raise ValueError(f"vertex index out of range: v={v}, n={self.n}")
        if not (0 <= c < self.k):
            raise ValueError(f"color index out of range: c={c}, k={self.k}")
        return self.x[v * self.k + c]


@dataclass(frozen=True)
class GraphColoringInstance:
    """Convenience container (optional)."""
    n_vertices: int
    edges: List[Tuple[int, int]]
    k: int
    precolored: Dict[int, int] | None = None
    enforce_exactly_one: bool = True


def build_graph_coloring_oracle(
    n_vertices: int,
    edges: List[Tuple[int, int]],
    k: int,
    precolored: Optional[Dict[int, int]] = None,
    enforce_exactly_one: bool = True,
) -> Tuple[QuantumCircuit, ColoringLayout]:
    """
    Build a **phase oracle** for the vertex graph coloring problem using one-hot encoding.

    Args:
      n_vertices: number of vertices (0..n_vertices-1)
      edges: list of undirected edges (u,v). Include each undirected edge once.
      k: number of colors (0..k-1)
      precolored: optional dict {v: c} fixing vertex v to color c
      enforce_exactly_one: if True, enforce that each vertex chooses exactly one color

    Returns:
      (oracle, layout)
        - oracle: QuantumCircuit that flips phase (-1) on satisfying assignments.
        - layout: ColoringLayout to address variable qubits x[v,c].

    Register order inside returned circuit:
      - QuantumRegister 'x' (variables), size n_vertices*k
      - Ancilla registers (various), including 'flag' and 'work'
    """
    if n_vertices <= 0:
        raise ValueError("n_vertices must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    precolored = precolored or {}

    # Validate edges
    for (u, v) in edges:
        if not (0 <= u < n_vertices and 0 <= v < n_vertices):
            raise ValueError(f"edge ({u},{v}) has vertex out of range (n={n_vertices})")
        if u == v:
            raise ValueError(f"self-loop edge ({u},{v}) is not supported for coloring")

    # Validate precolored
    for vv, cc in precolored.items():
        if not (0 <= vv < n_vertices):
            raise ValueError(f"precolored vertex out of range: {vv}")
        if not (0 <= cc < k):
            raise ValueError(f"precolored color out of range: {cc} (k={k})")

    m_edges = len(edges)

    # --------- Variable register ----------
    x = QuantumRegister(n_vertices * k, "x")
    layout = ColoringLayout(n_vertices, k, x)

    # --------- Ancillas for vertex constraints (exactly-one) ----------
    n_pairs = k * (k - 1) // 2
    pair_viol = AncillaRegister(n_vertices * n_pairs, "pv") if enforce_exactly_one and k >= 2 else AncillaRegister(0, "pv")
    atleast1 = AncillaRegister(n_vertices, "a1") if enforce_exactly_one else AncillaRegister(0, "a1")
    pair_all_zero = AncillaRegister(n_vertices, "p0") if enforce_exactly_one and k >= 2 else AncillaRegister(0, "p0")
    v_ok = AncillaRegister(n_vertices, "vok") if enforce_exactly_one else AncillaRegister(0, "vok")

    # --------- Ancillas for edge constraints ----------
    e_viol = AncillaRegister(m_edges * k, "ev") if m_edges > 0 else AncillaRegister(0, "ev")
    e_ok = AncillaRegister(m_edges, "eok") if m_edges > 0 else AncillaRegister(0, "eok")

    # Final flag
    flag = AncillaRegister(1, "flag")

    # Work ancillas for multi-controlled-X (v-chain): need up to controls-2.
    precolor_literals = 0
    for vv, cstar in precolored.items():
        precolor_literals += 1
        precolor_literals += (k - 1)

    final_controls = (n_vertices if enforce_exactly_one else 0) + m_edges + precolor_literals
    max_controls = max(k, (n_pairs if enforce_exactly_one else 0), final_controls)
    work_size = max(0, max_controls - 2)
    work = AncillaRegister(work_size, "work") if work_size > 0 else AncillaRegister(0, "work")

    # Compute circuit: compute flag=1 iff all constraints satisfied
    compute = QuantumCircuit(x, pair_viol, atleast1, pair_all_zero, v_ok, e_viol, e_ok, flag, work, name="compute")

    def _mcx(controls, target):
        """MCX wrapper: use v-chain if enough work ancillas, else fall back."""
        if len(controls) == 0:
            raise ValueError("mcx: empty controls")
        if len(controls) == 1:
            compute.cx(controls[0], target)
            return
        need = max(0, len(controls) - 2)
        if need <= len(work):
            compute.mcx(controls, target, ancilla_qubits=work[:need], mode="v-chain")
        else:
            compute.mcx(controls, target, mode="noancilla")

    # --------- Vertex constraints: ExactlyOne for each vertex ----------
    if enforce_exactly_one:
        for v in range(n_vertices):
            row = [layout.q(v, c) for c in range(k)]

            # atleast1[v] = 1 iff at least one in row is 1
            # compute all_zero into atleast1[v], then invert.
            for qb in row:
                compute.x(qb)
            if k == 1:
                compute.cx(row[0], atleast1[v])     # after X, 1 iff original 0
            else:
                _mcx(row, atleast1[v])              # 1 iff all were 0 (after X)
            for qb in row:
                compute.x(qb)
            compute.x(atleast1[v])                  # invert -> 1 iff at least one was 1

            if k >= 2:
                base = v * n_pairs

                # pair_viol = AND of each pair (detect >=2 ones)
                for idx, (c1, c2) in enumerate(combinations(range(k), 2)):
                    compute.ccx(layout.q(v, c1), layout.q(v, c2), pair_viol[base + idx])

                # pair_all_zero[v] = 1 iff all pair_viol are 0 (=> at most one)
                pbits = [pair_viol[base + i] for i in range(n_pairs)]
                for qb in pbits:
                    compute.x(qb)
                _mcx(pbits, pair_all_zero[v]) if n_pairs > 0 else compute.x(pair_all_zero[v])
                for qb in pbits:
                    compute.x(qb)

                # v_ok[v] = atleast1[v] AND pair_all_zero[v]
                compute.ccx(atleast1[v], pair_all_zero[v], v_ok[v])
            else:
                # k==1: exactly-one is x[v,0]==1
                compute.cx(layout.q(v, 0), v_ok[v])

    # --------- Edge constraints: adjacent vertices not same color ----------
    for e, (u, v) in enumerate(edges):
        for c in range(k):
            compute.ccx(layout.q(u, c), layout.q(v, c), e_viol[e * k + c])

        bits = [e_viol[e * k + c] for c in range(k)]
        for qb in bits:
            compute.x(qb)
        _mcx(bits, e_ok[e]) if k > 0 else compute.x(e_ok[e])
        for qb in bits:
            compute.x(qb)

    # --------- Final AND into flag ----------
    final_ctrls = []
    if enforce_exactly_one:
        final_ctrls += [v_ok[v] for v in range(n_vertices)]
    final_ctrls += [e_ok[e] for e in range(m_edges)]

    # Precolor literals: x[v,c*]==1 and x[v,c!=c*]==0
    zero_literals = []
    one_literals = []
    for vv, cstar in precolored.items():
        one_literals.append(layout.q(vv, cstar))
        for c in range(k):
            if c != cstar:
                zero_literals.append(layout.q(vv, c))

    final_ctrls += one_literals
    for qb in zero_literals:
        compute.x(qb)  # "==0" -> "==1" control
    final_ctrls += zero_literals

    if len(final_ctrls) == 0:
        compute.x(flag[0])  # no constraints -> always satisfied
    else:
        _mcx(final_ctrls, flag[0])

    for qb in zero_literals:
        compute.x(qb)

    # --------- Wrap as phase oracle: compute -> Z(flag) -> uncompute ----------
    oracle = QuantumCircuit(x, pair_viol, atleast1, pair_all_zero, v_ok, e_viol, e_ok, flag, work, name="oracle")
    oracle.compose(compute, inplace=True)
    oracle.z(flag[0])
    oracle.compose(compute.inverse(), inplace=True)

    return oracle, layout
