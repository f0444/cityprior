import numpy as np

from cityprior.pneuma import EVERY, LAT0, LON0, parse


def _line(track, typ, n):
    groups = []
    for i in range(n):
        lat = LAT0 + 1e-5 * i          # moving north ~1.1 m per 0.04 s step
        groups += [f"{lat:.7f}", f"{LON0:.7f}", "36.0", "0.1", "0.0", f"{0.04 * i:.2f}"]
    return "; ".join([str(track), typ, "10.0", "36.0"] + groups) + ";"


def test_parse_reads_repeated_groups_downsamples_and_projects():
    header = "track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time"
    text = "\n".join([header, _line(7, "Car", 50), _line(8, "Taxi", 20)])
    df = parse(text, file_id=3)
    assert set(df["track"]) == {300007, 300008}
    car = df[df["track"] == 300007]
    assert len(car) == int(np.ceil(50 / EVERY))
    assert np.allclose(np.diff(car["t"]), 0.04 * EVERY)
    assert np.allclose(car["speed"], 10.0)                       # 36 km/h
    assert abs(car["x"].iloc[0]) < 0.01                          # at the origin longitude
    assert np.allclose(car["heading"].dropna(), np.pi / 2, atol=0.05)  # heading north


def test_parse_keeps_the_valid_prefix_of_a_corrupted_row():
    header = "track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time"
    good = _line(1, "Car", 30)
    broken = _line(2, "Car", 30).replace(f"{0.04 * 12:.2f}", "Bicycle.0064", 1)
    df = parse("\n".join([header, good, broken]), file_id=0)
    assert len(df[df["track"] == 2]) == int(np.ceil(12 / EVERY))   # samples before the bad token survive
