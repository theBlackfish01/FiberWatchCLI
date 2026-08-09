import numpy as np

from phi_research.shift_forensics_v1 import _bh_adjust, _effect_rows


def test_bh_adjust_is_bounded_and_monotonic_by_rank():
    values = np.asarray([0.04, 0.001, 0.2, 0.02])
    adjusted = _bh_adjust(values)
    assert np.all((adjusted >= values) & (adjusted <= 1.0))
    order = np.argsort(values)
    assert np.all(np.diff(adjusted[order]) >= 0)


def test_effect_rows_are_deterministic_and_directional():
    target = np.asarray([[2.0, 1.0], [4.0, 1.5], [3.0, 2.0]])
    reference = np.asarray([[0.0, 1.0], [1.0, 1.1], [0.5, 1.2]])
    kwargs = {
        "feature_names": np.asarray(["amplitude_global_mean", "dynamics_delta_rms"]),
        "feature_groups": {"amplitude": ["amplitude_"], "dynamics": ["dynamics_"]},
        "draws": 200,
    }
    first = _effect_rows(target, reference, rng=np.random.default_rng(7), **kwargs)
    second = _effect_rows(target, reference, rng=np.random.default_rng(7), **kwargs)
    assert first == second
    assert first[0]["standardized_effect"] > 0
    assert first[0]["feature_group"] == "amplitude"
