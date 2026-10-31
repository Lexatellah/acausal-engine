import numpy as np
import pytest
from acausal_engine.quantum_switch import (
    quantum_switch_2qubit,
    fixed_order_process,
)


def test_quantum_switch_is_valid_process():
    W = quantum_switch_2qubit()
    assert W.dim == 8
    assert np.isclose(W.trace(), 1.0, atol=1e-9)
    eigvals = np.linalg.eigvalsh(W.matrix)
    assert np.all(eigvals >= -1e-9)


def test_quantum_switch_is_pure():
    W = quantum_switch_2qubit()
    rank = np.linalg.matrix_rank(W.matrix, tol=1e-9)
    assert rank == 1, f"Expected rank 1, got {rank}"


def test_fixed_order_has_no_control_coherence():
    W_ab = fixed_order_process("AB")
    W_ba = fixed_order_process("BA")
    assert not np.allclose(W_ab.matrix, W_ba.matrix)

    W_switch = quantum_switch_2qubit()
    d_target = 4
    block_01_switch = W_switch.matrix[:d_target, d_target : 2 * d_target]
    assert np.linalg.norm(block_01_switch) > 1e-6, (
        "Quantum switch should have coherence between control states."
    )
    block_01_fixed = W_ab.matrix[:d_target, d_target : 2 * d_target]
    assert np.linalg.norm(block_01_fixed) < 1e-9, (
        "Fixed order should have no coherence between control states."
    )


def test_invalid_order_raises():
    with pytest.raises(ValueError):
        fixed_order_process("AC")

def test_qudit_switch_dim_3():
    from acausal_engine.quantum_switch import quantum_switch_qudit
    W = quantum_switch_qudit(target_dim=3)
    assert W.dim == 18  # 2 * 3 * 3
    assert np.isclose(W.trace(), 1.0, atol=1e-9)
    rank = np.linalg.matrix_rank(W.matrix, tol=1e-9)
    assert rank == 1, f"Expected rank 1, got {rank}"


def test_qudit_switch_invalid_dim():
    from acausal_engine.quantum_switch import quantum_switch_qudit
    import pytest
    with pytest.raises(ValueError):
        quantum_switch_qudit(target_dim=0)


def test_qudit_switch_off_diagonal_coherence():
    from acausal_engine.quantum_switch import quantum_switch_qudit
    for d in (2, 3, 4):
        W = quantum_switch_qudit(target_dim=d)
        d_target_block = d * d
        block = W.matrix[:d_target_block, d_target_block : 2 * d_target_block]
        assert np.linalg.norm(block) > 1e-6, f"d={d}: switch should have coherence."
