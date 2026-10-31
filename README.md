# Acausal Engine

A classical simulation toolkit for **indefinite causal order**.

![CI](https://github.com/Lexatellah/acausal-engine/actions/workflows/test.yml/badge.svg)

> **Honesty note:** This is not a physical acausal computer. It simulates the mathematical objects in Shmueli (2026). The complexity results are conditional and not reproduced here.

## Install

    pip install git+https://github.com/Lexatellah/acausal-engine

## Demo

    python examples/switch_vs_fixed.py

Switch coherence **0.5** vs fixed-order **0.0** -- the numerical fingerprint of indefinite causal order.

## Tests

    python -m pytest tests/ -v

12 tests, green on Windows and Ubuntu (Python 3.10 / 3.11 / 3.12).

## What is inside

- `acausal_engine/process_matrix.py` -- Choi matrix wrapper with validation
- `acausal_engine/quantum_switch.py` -- 2-qubit and qudit quantum switch
- `acausal_engine/uso.py` -- unique-sink orientation on hypercubes

## License

MIT.