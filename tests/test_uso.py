import pytest
from acausal_engine.uso import RandomUSO


def test_random_uso_dim_1():
    uso = RandomUSO(dim=1, seed=42)
    assert uso.dim == 1
    assert len(uso.orientation) == 2
    sink = uso.find_sink_bruteforce()
    assert uso.is_sink(sink)


def test_random_uso_dim_2():
    uso = RandomUSO(dim=2, seed=42)
    assert len(uso.orientation) == 4
    sink = uso.find_sink_bruteforce()
    assert uso.is_sink(sink)
    assert uso.find_sink_walk() == sink


def test_random_uso_dim_3():
    uso = RandomUSO(dim=3, seed=42)
    assert len(uso.orientation) == 8
    sink = uso.find_sink_bruteforce()
    assert uso.is_sink(sink)
    assert uso.find_sink_walk() == sink


def test_multiple_seeds_produce_valid_usos():
    for seed in range(10):
        uso = RandomUSO(dim=3, seed=seed)
        sink = uso.find_sink_bruteforce()
        assert uso.is_sink(sink)


def test_walk_terminates_on_valid_uso():
    uso = RandomUSO(dim=4, seed=123)
    sink = uso.find_sink_walk()
    assert uso.is_sink(sink)
