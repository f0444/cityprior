import numpy as np

from cityprior.dlr_experiments import _mi_columns, governing_signals


def test_mutual_information_is_zero_for_independent_and_one_bit_for_identical():
    rng = np.random.default_rng(0)
    y = rng.random(20_000) < 0.5
    g = np.stack([y, rng.random(20_000) < 0.5], axis=1)
    mi = _mi_columns(g, y)
    assert abs(mi[0] - 1.0) < 0.01 and mi[1] < 0.001


def test_governing_signal_is_learned_from_stops():
    """Place 1 obeys light 2, place 2 obeys light 0; light 1 is noise."""
    rng = np.random.default_rng(1)
    n = 3000
    place = np.repeat([1, 2], n)
    green = rng.random((2 * n, 3)) < 0.4
    stopped = np.where(place == 1, ~green[:, 2], ~green[:, 0])
    stopped ^= rng.random(2 * n) < 0.05                      # a few vehicles run the light or stop anyway
    gov = governing_signals(place, green, stopped)
    assert gov[1][0] == 2 and gov[2][0] == 0


def test_place_without_a_light_gets_none():
    rng = np.random.default_rng(2)
    place = np.full(2000, 7)
    green = rng.random((2000, 4)) < 0.5
    stopped = rng.random(2000) < 0.3
    assert 7 not in governing_signals(place, green, stopped)
