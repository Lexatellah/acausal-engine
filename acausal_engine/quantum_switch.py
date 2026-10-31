"""
Quantum switch process matrices.

Two constructions:
    quantum_switch_2qubit()  - the canonical d=2 case (explicit)
    quantum_switch_qudit(d)  - general target dimension d
"""

import numpy as np
from .process_matrix import ChoiMatrix


def _ket0() -> np.ndarray:
    return np.array([1, 0], dtype=complex)


def _ket1() -> np.ndarray:
    return np.array([0, 1], dtype=complex)


def _swap_gate(dim: int) -> np.ndarray:
    """SWAP gate for two systems of dimension `dim`."""
    d2 = dim * dim
    S = np.zeros((d2, d2), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            S[i * dim + j, j * dim + i] = 1.0
    return S


def _phi_plus(d: int) -> np.ndarray:
    """Maximally entangled state on C^d x C^d."""
    phi = np.zeros((d * d,), dtype=complex)
    for i in range(d):
        phi[i * d + i] = 1.0 / np.sqrt(d)
    return phi


def quantum_switch_qudit(target_dim: int = 2) -> ChoiMatrix:
    """
    Choi matrix of the quantum switch for target dimension `target_dim`.

    Control is always 2-dimensional (|0> = A before B, |1> = B before A).
    Total Hilbert dimension = 2 * d * d.

    Parameters
    ----------
    target_dim : int
        Dimension of each target system (default 2).

    Returns
    -------
    ChoiMatrix
    """
    d = int(target_dim)
    if d < 1:
        raise ValueError("target_dim must be >= 1.")

    d_control = 2
    d_total = d_control * d * d

    phi = _phi_plus(d)
    swap = _swap_gate(d)
    phi_swapped = swap @ phi

    w = np.zeros((d_total,), dtype=complex)
    w[0 : d * d] = phi / np.sqrt(2)
    w[d * d : 2 * d * d] = phi_swapped / np.sqrt(2)

    w = w.reshape(d_total, 1)
    W = w @ w.conj().T
    W = W / np.trace(W)
    return ChoiMatrix(W)


def quantum_switch_2qubit() -> ChoiMatrix:
    """Canonical 2-qubit switch (d=2). Alias of quantum_switch_qudit(2)."""
    return quantum_switch_qudit(2)


def fixed_order_process(order: str = "AB") -> ChoiMatrix:
    """
    Fixed-order process matrix on control (2-dim) and two target qubits.
    order = 'AB' or 'BA'.
    """
    if order not in ("AB", "BA"):
        raise ValueError("order must be 'AB' or 'BA'.")

    d_target = 2
    d_total = 2 * d_target * d_target

    phi = _phi_plus(d_target)
    swap = _swap_gate(d_target)
    phi_swapped = swap @ phi

    if order == "AB":
        target_state = phi
        control_state = _ket0()
    else:
        target_state = phi_swapped
        control_state = _ket1()

    w = np.kron(control_state, target_state).reshape(d_total, 1)
    W = w @ w.conj().T
    W = W / np.trace(W)
    return ChoiMatrix(W)
