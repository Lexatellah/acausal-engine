"""
Acausal Engine: A classical simulation toolkit for indefinite causal order.

Based on:
    Shmueli, O. (2026). "What Can Be Computed with Indefinite Causal Structure?"
"""

__version__ = "0.1.0"

from .process_matrix import ChoiMatrix
from .quantum_switch import quantum_switch_2qubit, fixed_order_process
from .uso import HypercubeUSO, RandomUSO

__all__ = [
    "ChoiMatrix",
    "quantum_switch_2qubit",
    "fixed_order_process",
    "HypercubeUSO",
    "RandomUSO",
]
