"""
Demonstrate the difference between the quantum switch and fixed-order
processes. The switch has coherence between control states |0> and |1>;
fixed orders do not. That coherence is the signature of indefinite
causal order.

Run: python examples/switch_vs_fixed.py
"""

import numpy as np
from acausal_engine.quantum_switch import (
    quantum_switch_2qubit,
    fixed_order_process,
)


def main() -> None:
    W_switch = quantum_switch_2qubit()
    W_ab = fixed_order_process("AB")
    W_ba = fixed_order_process("BA")

    print("=== Choi matrix traces (all should be 1.0) ===")
    for name, W in [("switch", W_switch), ("AB", W_ab), ("BA", W_ba)]:
        print(f"  {name:>8s}: trace = {W.trace():.6f}  dim = {W.dim}")

    print()
    print("=== Off-diagonal control block norms ===")
    d_target = 4  # 2 target qubits -> d^2 = 4
    for name, W in [("switch", W_switch), ("AB", W_ab), ("BA", W_ba)]:
        block = W.matrix[:d_target, d_target : 2 * d_target]
        print(f"  {name:>8s}: ||block|| = {np.linalg.norm(block):.8f}")

    print()
    print("Interpretation:")
    print("  The switch has non-zero control coherence.")
    print("  Fixed-order processes have exactly zero.")
    print("  => the switch is NOT a convex mixture of fixed orders.")
    print("  That is indefinite causal order, computed on your laptop.")


if __name__ == "__main__":
    main()
