from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit_aer import AerSimulator

from graph_coloring2 import build_graph_coloring_oracle_iqm_safe

BackendKind = Literal["sim", "iqm"]


@dataclass(frozen=True)
class GroverRunConfig:
    backend_kind: BackendKind = "sim"
    shots: int = 1024
    iters: int = 1
    optimization_level: int = 1
    decompose_reps: int = 5

    # simulator
    sim_method: str = "matrix_product_state"

    # IQM
    iqm_server_url: Optional[str] = None
    iqm_backend_name: Optional[str] = None


def build_state_preparation_x_only(oracle: QuantumCircuit, x_len: int) -> QuantumCircuit:
    A = QuantumCircuit(*oracle.qregs, name="A")
    A.h(A.qubits[:x_len])
    return A


def build_grover_circuit(
    n: int,
    edges: List[Tuple[int, int]],
    k: int,
    precolored: Optional[Dict[int, int]] = None,
    iters: int = 1,
) -> Tuple[QuantumCircuit, int]:
    precolored = precolored or {}

    oracle, layout = build_graph_coloring_oracle_iqm_safe(
        n_vertices=n,
        edges=edges,
        k=k,
        precolored=precolored,
        enforce_exactly_one=True,
    )

    x_len = n * k
    A = build_state_preparation_x_only(oracle, x_len)
    reflection_qubits = list(range(x_len))

    Q = grover_operator(
        oracle=oracle,
        state_preparation=A,
        reflection_qubits=reflection_qubits,
    )

    c = ClassicalRegister(x_len, "c")
    qc = QuantumCircuit(*oracle.qregs, c, name="grover_run")
    qc.compose(A, inplace=True)
    for _ in range(iters):
        qc.compose(Q, inplace=True)

    qc.measure(layout.x, c)
    return qc, x_len


def _get_backend(config: GroverRunConfig):
    if config.backend_kind == "sim":
        return AerSimulator(method=config.sim_method)

    if config.backend_kind == "iqm":
        if not config.iqm_server_url or not config.iqm_backend_name:
            raise ValueError("IQM実機では iqm_server_url と iqm_backend_name が必要です。")
        # 遅延 import（sim 環境で iqm が無くても落ちない）
        from iqm.qiskit_iqm.iqm_provider import IQMProvider

        provider = IQMProvider(url=config.iqm_server_url)
        return provider.get_backend(config.iqm_backend_name)

    raise ValueError(f"Unknown backend_kind: {config.backend_kind}")


def run_grover(
    n: int,
    edges: List[Tuple[int, int]],
    k: int,
    precolored: Optional[Dict[int, int]] = None,
    config: Optional[GroverRunConfig] = None,
):
    config = config or GroverRunConfig()

    qc, _ = build_grover_circuit(n=n, edges=edges, k=k, precolored=precolored, iters=config.iters)
    qc2 = qc.decompose(reps=config.decompose_reps)

    backend = _get_backend(config)
    opt_level = max(1, config.optimization_level) if config.backend_kind == "iqm" else config.optimization_level

    tqc = transpile(qc2, backend=backend, optimization_level=opt_level)
    job = backend.run(tqc, shots=config.shots)
    counts = job.result().get_counts()
    return counts, tqc