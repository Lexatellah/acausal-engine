import numpy as np
import pytest
from acausal_engine.quantum_switch import (
    quantum_switch_2qubit,
    quantum_switch_qudit,
    fixed_order_process,
)

# Cross-platform numerical tolerance. Ubuntu BLAS and Windows BLAS differ
# at the 1e-12 level; use rtol for relative, atol for absolute.
RTOL = 1e-6
ATOL = 1e-6


def _is_pure(matrix: np.ndarray) -> bool:
    """A pure density matrix has exactly one non-negligible singular value."""
    s = np.linalg.svd(matrix, compute_uv=False)
    # Largest singular value should be ~1 (trace-normalized), rest ~0
    return bool(s[0] > 0.99 and np.all(s[1:] < 1e-3))


def test_quantum_switch_is_valid_process():
    W = quantum_switch_2qubit()
    assert W.dim == 8
    assert np.isclose(W.trace(), 1.0, rtol=RTOL, atol=ATOL)
    assert np.allclose(W.matrix, W.matrix.conj().T, rtol=RTOL, atol=ATOL)
    eigvals = np.linalg.eigvalsh(W.matrix)
    assert np.all(eigvals > -ATOL), f"Min eigenvalue {eigvals.min():.3e}"


def test_quantum_switch_is_pure():
    W = quantum_switch_2qubit()
    assert _is_pure(W.matrix), "Quantum switch Choi matrix should be pure (rank 1)"


def test_fixed_order_has_no_control_coherence():
    W_ab = fixed_order_process("AB")
    W_ba = fixed_order_process("BA")
    assert not np.allclose(W_ab.matrix, W_ba.matrix, rtol=RTOL, atol=ATOL)

    W_switch = quantum_switch_2qubit()
    d_target = 4
    block_switch = W_switch.matrix[:d_target, d_target : 2 * d_target]
    block_fixed = W_ab.matrix[:d_target, d_target : 2 * d_target]

    # Real switch coherence ~0.5; fixed order ~0. Use wide margins.
    assert np.linalg.norm(block_switch) > 1e-2, (
        "Quantum switch should have coherence between control states."
    )
    assert np.linalg.norm(block_fixed) < 1e-4, (
        "Fixed order should have negligible coherence between control states."
    )


def test_invalid_order_raises():
    with pytest.raises(ValueError):
        fixed_order_process("AC")


def test_qudit_switch_dim_3():
    W = quantum_switch_qudit(target_dim=3)
    assert W.dim == 18
    assert np.isclose(W.trace(), 1.0, rtol=RTOL, atol=ATOL)
    assert _is_pure(W.matrix), "d=3 switch should be pure"


def test_qudit_switch_invalid_dim():
    with pytest.raises(ValueError):
        quantum_switch_qudit(target_dim=0)


def test_qudit_switch_off_diagonal_coherence():
    for d in (2, 3, 4):
        W = quantum_switch_qudit(target_dim=d)
        d2 = d * d
        block = W.matrix[:d2, d2 : 2 * d2]
        assert np.linalg.norm(block) > 1e-2, (
            f"d={d}: switch should have coherence between control states."
        )
