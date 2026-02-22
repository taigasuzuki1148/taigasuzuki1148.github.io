from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Optional

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister


@dataclass(frozen=True)
class ColoringLayout:
    n: int
    k: int
    x: QuantumRegister

    def q(self, v: int, c: int):
        if not (0 <= v < self.n):
            raise ValueError(f"vertex index out of range: v={v}, n={self.n}")
        if not (0 <= c < self.k):
            raise ValueError(f"color index out of range: c={c}, k={self.k}")
        return self.x[v * self.k + c]


def build_graph_coloring_oracle_iqm_safe(
    n_vertices: int,
    edges: List[Tuple[int, int]],
    k: int,
    precolored: Optional[Dict[int, int]] = None,
    enforce_exactly_one: bool = True,
) -> Tuple[QuantumCircuit, ColoringLayout]:
    """
    IQM実機でのトランスパイル/実行を安定させるために
      - サイズ0レジスタを回路に含めない
      - MCXはv-chainのみ（work ancillaを確実に確保）
    に寄せた phase oracle.
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

    # --------- Sizes for ancillas ----------
    n_pairs = k * (k - 1) // 2

    need_vertex = enforce_exactly_one
    need_pairs = need_vertex and k >= 2 and n_pairs > 0
    need_edges = m_edges > 0

    # Precolor literals count (controls): x[v,c*]==1 and x[v,others]==0
    precolor_literals = 0
    for vv, cstar in precolored.items():
        precolor_literals += 1          # the "1" literal
        precolor_literals += (k - 1)    # the "0" literals

    # Worst-case controls that will be fed into MCX in this circuit:
    # - atleast1: k controls (after X)
    # - pair_all_zero: n_pairs controls
    # - final AND: n_vertices + m_edges + precolor_literals controls (depending on flags)
    final_controls = (n_vertices if enforce_exactly_one else 0) + m_edges + precolor_literals
    max_controls = max(
        (k if need_vertex else 0),
        (n_pairs if need_pairs else 0),
        final_controls,
        2,  # keep non-negative
    )
    work_size = max(0, max_controls - 2)

    # --------- Ancilla registers (only if size>0) ----------
    regs = [x]

    pair_viol = AncillaRegister(n_vertices * n_pairs, "pv") if need_pairs else None
    atleast1 = AncillaRegister(n_vertices, "a1") if need_vertex else None
    pair_all_zero = AncillaRegister(n_vertices, "p0") if need_pairs else None
    v_ok = AncillaRegister(n_vertices, "vok") if need_vertex else None

    e_viol = AncillaRegister(m_edges * k, "ev") if need_edges else None
    e_ok = AncillaRegister(m_edges, "eok") if need_edges else None

    flag = AncillaRegister(1, "flag")
    work = AncillaRegister(work_size, "work") if work_size > 0 else None

    for r in [pair_viol, atleast1, pair_all_zero, v_ok, e_viol, e_ok, flag, work]:
        if r is not None and len(r) > 0:
            regs.append(r)

    compute = QuantumCircuit(*regs, name="compute")

    def _mcx_vchain(controls, target):
        """MCX wrapper: always v-chain (IQM向けに挙動を固定)."""
        if len(controls) == 0:
            raise ValueError("mcx: empty controls")
        if len(controls) == 1:
            compute.cx(controls[0], target)
            return
        need = max(0, len(controls) - 2)
        if work is None or need > len(work):
            raise ValueError(
                f"Not enough work ancillas for v-chain MCX: "
                f"controls={len(controls)} need_work={need} have={0 if work is None else len(work)}"
            )
        compute.mcx(controls, target, ancilla_qubits=work[:need], mode="v-chain")

    # --------- Vertex constraints: ExactlyOne for each vertex ----------
    if enforce_exactly_one:
        assert atleast1 is not None and v_ok is not None

        for v in range(n_vertices):
            row = [layout.q(v, c) for c in range(k)]

            # atleast1[v] = 1 iff at least one in row is 1
            for qb in row:
                compute.x(qb)
            if k == 1:
                compute.cx(row[0], atleast1[v])  # after X, 1 iff original 0
            else:
                _mcx_vchain(row, atleast1[v])     # 1 iff all were 0 (after X)
            for qb in row:
                compute.x(qb)
            compute.x(atleast1[v])               # invert -> 1 iff at least one was 1

            if k >= 2:
                assert pair_viol is not None and pair_all_zero is not None

                base = v * n_pairs

                # pair_viol = AND of each pair (detect >=2 ones)
                for idx, (c1, c2) in enumerate(combinations(range(k), 2)):
                    compute.ccx(layout.q(v, c1), layout.q(v, c2), pair_viol[base + idx])

                # pair_all_zero[v] = 1 iff all pair_viol are 0  (=> at most one)
                pbits = [pair_viol[base + i] for i in range(n_pairs)]
                for qb in pbits:
                    compute.x(qb)
                _mcx_vchain(pbits, pair_all_zero[v]) if n_pairs > 0 else compute.x(pair_all_zero[v])
                for qb in pbits:
                    compute.x(qb)

                # v_ok[v] = atleast1[v] AND pair_all_zero[v]
                compute.ccx(atleast1[v], pair_all_zero[v], v_ok[v])
            else:
                # k==1: exactly-one is x[v,0]==1
                compute.cx(layout.q(v, 0), v_ok[v])

    # --------- Edge constraints: adjacent vertices not same color ----------
    if m_edges > 0:
        assert e_viol is not None and e_ok is not None

        for e, (u, v) in enumerate(edges):
            for c in range(k):
                compute.ccx(layout.q(u, c), layout.q(v, c), e_viol[e * k + c])

            bits = [e_viol[e * k + c] for c in range(k)]
            for qb in bits:
                compute.x(qb)
            _mcx_vchain(bits, e_ok[e])  # k>0 guaranteed
            for qb in bits:
                compute.x(qb)

    # --------- Final AND into flag ----------
    final_ctrls = []
    if enforce_exactly_one:
        assert v_ok is not None
        final_ctrls += [v_ok[v] for v in range(n_vertices)]
    if m_edges > 0:
        assert e_ok is not None
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
        _mcx_vchain(final_ctrls, flag[0])

    for qb in zero_literals:
        compute.x(qb)

    # --------- Wrap as phase oracle: compute -> Z(flag) -> uncompute ----------
    oracle = QuantumCircuit(*regs, name="oracle")
    oracle.compose(compute, inplace=True)
    oracle.z(flag[0])
    oracle.compose(compute.inverse(), inplace=True)

    return oracle, layout