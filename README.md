# Acausal Engine

A classical simulation toolkit for exploring **indefinite causal order** within the process matrix formalism.

![CI](https://github.com/Lexatellah/acausal-engine/actions/workflows/test.yml/badge.svg)

> **Honesty note:** This is not a physical acausal computer. It simulates the mathematical objects in Shmueli (2026), "What Can Be Computed with Indefinite Causal Structure?" The complexity results are conditional and not reproduced here.

## Install

    pip install git+https://github.com/Lexatellah/acausal-engine

## Demo

    python examples/switch_vs_fixed.py

Output shows switch coherence 0.5 vs fixed-order coherence 0.0.

## Tests

    python -m pytest tests/ -v

12 tests, green on Windows and Ubuntu (Python 3.10 / 3.11 / 3.12).

## What is inside

- acausal_engine/process_matrix.py -- Choi matrix wrapper with Hermitian + PSD validation
- acausal_engine/quantum_switch.py -- 2-qubit and general qudit quantum switch
- acausal_engine/uso.py -- unique-sink orientation on hypercubes

## License

MIT.
