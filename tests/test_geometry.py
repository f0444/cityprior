import numpy as np

from cityprior.geometry import Grid, segment_hits_boxes, wrap_angle


def test_grid_indexing_and_accumulate():
    g = Grid(x0=0.0, y0=0.0, nx=10, ny=5, cell=1.0)
    assert g.flat(np.array([0.5]), np.array([0.5]))[0] == 0
    assert g.flat(np.array([9.5]), np.array([4.5]))[0] == 49
    assert g.flat(np.array([10.5]), np.array([0.5]))[0] == -1
    acc = g.accumulate(np.array([1.2, 1.7, 3.3]), np.array([2.1, 2.9, 0.1]))
    assert acc[2, 1] == 2 and acc[0, 3] == 1 and acc.sum() == 3


def test_segment_box_intersection():
    box = dict(centers=np.array([[5.0, 0.0]]), headings=np.array([0.0]),
               lengths=np.array([4.0]), widths=np.array([2.0]))
    through = segment_hits_boxes(np.array([[0.0, 0.0]]), np.array([[10.0, 0.0]]), **box)
    beside = segment_hits_boxes(np.array([[0.0, 2.0]]), np.array([[10.0, 2.0]]), **box)
    short = segment_hits_boxes(np.array([[0.0, 0.0]]), np.array([[2.5, 0.0]]), **box)
    assert through[0] and not beside[0] and not short[0]


def test_segment_rotated_box():
    # 4 m x 0.5 m box rotated 90 degrees: blocks a horizontal ray at y=1.5
    kw = dict(centers=np.array([[5.0, 0.0]]), headings=np.array([np.pi / 2]),
              lengths=np.array([4.0]), widths=np.array([0.5]))
    assert segment_hits_boxes(np.array([[0.0, 1.5]]), np.array([[10.0, 1.5]]), **kw)[0]
    assert not segment_hits_boxes(np.array([[0.0, 2.5]]), np.array([[10.0, 2.5]]), **kw)[0]


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
