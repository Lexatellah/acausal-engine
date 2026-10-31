"""
Explicit construction of the 2-qubit quantum switch process matrix.
"""

import numpy as np
from .process_matrix import ChoiMatrix


def _I(dim: int) -> np.ndarray:
    return np.eye(dim, dtype=complex)


def _ket0() -> np.ndarray:
    return np.array([1, 0], dtype=complex)


def _ket1() -> np.ndarray:
    return np.array([0, 1], dtype=complex)


def _swap_gate(dim: int) -> np.ndarray:
    d2 = dim * dim
    S = np.zeros((d2, d2), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            S[i * dim + j, j * dim + i] = 1.0
    return S


def quantum_switch_2qubit() -> ChoiMatrix:
    """
    Construct the Choi matrix of the 2-qubit quantum switch.

    Basis: |c>|a>|b> with c in {0,1} (control), a,b in {0,1} (targets).
    The pure state |w> is:
        (|0>_C |Phi+>_AB + |1>_C |Phi+>_BA) / sqrt(2)
    where |Phi+> = (|00> + |11>)/sqrt(2).
    """
    d_control = 2
    d_target = 2
    d_total = d_control * d_target * d_target  # 8

    phi_plus = np.zeros((d_target * d_target,), dtype=complex)
    phi_plus[0 * d_target + 0] = 1.0 / np.sqrt(2)
    phi_plus[1 * d_target + 1] = 1.0 / np.sqrt(2)

    swap = _swap_gate(d_target)
    phi_plus_swapped = swap @ phi_plus

    w = np.zeros((d_total,), dtype=complex)
    w[0 * (d_target * d_target) : 1 * (d_target * d_target)] = (
        phi_plus / np.sqrt(2)
    )
    w[1 * (d_target * d_target) : 2 * (d_target * d_target)] = (
        phi_plus_swapped / np.sqrt(2)
    )

    w = w.reshape(d_total, 1)
    W = w @ w.conj().T
    W = W / np.trace(W)
    return ChoiMatrix(W)


def fixed_order_process(order: str = "AB") -> ChoiMatrix:
    """
    Construct a trivial fixed-order process matrix for comparison.
    order = 'AB' or 'BA'.
    """
    d_control = 2
    d_target = 2
    d_total = d_control * d_target * d_target  # 8

    if order not in ("AB", "BA"):
        raise ValueError("order must be 'AB' or 'BA'.")

    phi_plus = np.zeros((d_target * d_target,), dtype=complex)
    phi_plus[0 * d_target + 0] = 1.0 / np.sqrt(2)
    phi_plus[1 * d_target + 1] = 1.0 / np.sqrt(2)

    swap = _swap_gate(d_target)
    phi_plus_swapped = swap @ phi_plus

    if order == "AB":
        target_state = phi_plus
        control_state = _ket0()
    else:
        target_state = phi_plus_swapped
        control_state = _ket1()

    w = np.kron(control_state, target_state).reshape(d_total, 1)
    W = w @ w.conj().T
    W = W / np.trace(W)
    return ChoiMatrix(W)
