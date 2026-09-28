import numpy as np

from cityprior.sim.engine import Infra, sample_encounters, simulate
from cityprior.sim.planner import camera_coverage
from cityprior.sim.world import Profile, SimParams

P = SimParams()
FAST = np.full(P.grid.shape, P.v_limit)
COV = camera_coverage(1.0, P)


def _truth() -> Profile:
    x = np.arange(0.0, P.street_len + 1e-9, 1.0)
    return Profile(x, np.full(len(x), 50 / 3600 / P.street_len))


def _enc(n=800, seed=0, tau=None):
    e = sample_encounters(np.random.default_rng(seed), n, P, np.full(101, 1.5), attentive=False)
    if tau is not None:
        e.tau[:] = tau
    return e


def test_ghosts_are_never_hit_or_seen_onboard():
    res = simulate(_enc(), _truth(), FAST, P, Infra(COV, True, ghost=True))
    assert res["collided"].sum() == 0 and res["near"].sum() == 0
    assert res["known_by_infra_first"].mean() > 0.3          # the camera did report them


def test_naive_car_brakes_for_ghosts_onboard_priority_limits_it():
    enc = _enc(tau=-1.5)                                        # phantom steps out just before the car arrives
    popup = dict(ghost=True, report_from_y=P.occlusion_edge - 0.3)
    naive = simulate(enc, _truth(), FAST, P, Infra(COV, True, **popup))
    guarded = simulate(enc, _truth(), FAST, P, Infra(COV, True, onboard_priority=True, **popup))
    assert naive["max_decel"].mean() > 2 * guarded["max_decel"].mean()


def test_onboard_priority_keeps_real_pedestrians_safe():
    enc = _enc(n=1500, seed=3)
    honest = simulate(enc, _truth(), FAST, P, Infra(COV, True))
    guarded = simulate(enc, _truth(), FAST, P, Infra(COV, True, onboard_priority=True))
    assert guarded["collided"].sum() <= honest["collided"].sum() + 2


def test_deletion_leaves_evidence_for_an_audit():
    enc = _enc(n=1500, seed=4)
    honest = simulate(enc, _truth(), FAST, P, Infra(COV, True))
    deleting = simulate(enc, _truth(), FAST, P, Infra(COV, True, delete_share=1.0))
    assert deleting["known_by_infra_first"].sum() == 0
    assert deleting["unreported_seen"].mean() > 10 * max(honest["unreported_seen"].mean(), 1e-3)
    assert deleting["collided"].sum() > honest["collided"].sum()
