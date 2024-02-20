import pytest
import numpy as np

from pupillometry.pupilLightResponse import PupilLightResponse


@pytest.fixture()
def plr() -> PupilLightResponse:
    return PupilLightResponse(
        time=np.arange(0, 15),
        size=np.array(
            [100, 101, 102, 103, 104, 98, 90, 91, 92, 93, 95, 97, 99, 105, 70]
        ),
        stimuli_start=4,
        stimuli_end=9,
    )


def test_initialize_pupil_size_time():
    with pytest.raises(ValueError):
        PupilLightResponse(
            time=np.array([0, 1, 2]),
            size=np.array([0, 1]),
            stimuli_start=0,
            stimuli_end=1,
        )

    with pytest.raises(ValueError):
        PupilLightResponse(
            time=np.array([0, 1]),
            size=np.array([0, 1, 2]),
            stimuli_start=0,
            stimuli_end=1,
        )


def test_initialize_pulse_start_end_valid_range():
    with pytest.raises(ValueError):
        PupilLightResponse(
            time=np.array([0, 1]),
            size=np.array([0, 1]),
            stimuli_start=0,
            stimuli_end=2,
        )
    with pytest.raises(ValueError):
        PupilLightResponse(
            time=np.array([0, 1]),
            size=np.array([0, 1]),
            stimuli_start=-1,
            stimuli_end=2,
        )
    with pytest.raises(ValueError):
        PupilLightResponse(
            time=np.array([0, 1]),
            size=np.array([0, 1]),
            stimuli_start=1,
            stimuli_end=1,
        )


def test_get_size(plr: PupilLightResponse):
    np.testing.assert_allclose(
        plr.get_size(),
        np.array(
            [100, 101, 102, 103, 104, 98, 90, 91, 92, 93, 95, 97, 99, 105, 70]
        ),
    )
    np.testing.assert_allclose(
        plr.get_size(end_time=5), np.array([100, 101, 102, 103, 104, 98])
    )
    np.testing.assert_allclose(
        plr.get_size(start_time=5),
        np.array([98, 90, 91, 92, 93, 95, 97, 99, 105, 70]),
    )
    np.testing.assert_allclose(
        plr.get_size(start_time=2, end_time=5), np.array([102, 103, 104, 98])
    )


def test_get_time(plr: PupilLightResponse):
    np.testing.assert_allclose(
        plr.get_time(),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    )
    np.testing.assert_allclose(
        plr.get_time(end_time=5), np.array([0, 1, 2, 3, 4, 5])
    )
    np.testing.assert_allclose(
        plr.get_time(start_time=5),
        np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    )
    np.testing.assert_allclose(
        plr.get_time(start_time=2, end_time=5), np.array([2, 3, 4, 5])
    )


def test_start_time(plr: PupilLightResponse):
    assert plr.get_start_time() == 0


def test_end_time(plr: PupilLightResponse):
    assert plr.get_end_time() == 14


def test_get_stimuli_start(plr: PupilLightResponse):
    assert plr.get_stimuli_start() == 4


def test_get_stimuli_end(plr: PupilLightResponse):
    assert plr.get_stimuli_end() == 9


def test_get_t_max_constriction(plr: PupilLightResponse):
    assert plr.get_t_max_constriction() == 6


def test_get_p0(plr: PupilLightResponse):
    time, size = plr.get_p0()
    np.testing.assert_allclose(time, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(size, np.array([100, 101, 102, 103, 104]))


def test_get_p1(plr: PupilLightResponse):
    time, size = plr.get_p1()
    np.testing.assert_allclose(time, np.array([4, 5, 6]))
    np.testing.assert_allclose(size, np.array([104, 98, 90]))


def test_get_p2(plr: PupilLightResponse):
    time, size = plr.get_p2()
    np.testing.assert_allclose(time, np.array([7, 8, 9]))
    np.testing.assert_allclose(size, np.array([91, 92, 93]))


def test_get_p3(plr: PupilLightResponse):
    time, size = plr.get_p3()
    np.testing.assert_allclose(time, np.array([9, 10, 11, 12, 13, 14]))
    np.testing.assert_allclose(size, np.array([93, 95, 97, 99, 105, 70]))


def test_get_baseline(plr: PupilLightResponse):
    assert plr.get_baseline(0, 1) == 100.5
    assert plr.get_baseline(0, 2) == 101
    assert plr.get_baseline(0, 3) == 101.5
    assert plr.get_baseline(0, 4) == 102
    assert plr.get_baseline(1, 4) == 102.5
    assert plr.get_baseline(2, 4) == 103
    assert plr.get_baseline(3, 4) == 103.5
    with pytest.raises(ValueError):
        plr.get_baseline(1, 0)
    with pytest.raises(ValueError):
        plr.get_baseline(-1, 0)
    with pytest.raises(ValueError):
        plr.get_baseline(0, 20)
    with pytest.raises(ValueError):
        plr.get_baseline(5, 20)


def test_normalize():
    plr = PupilLightResponse(
        time=np.array([0, 1, 2, 3, 4]),
        size=np.array([0, 1, 2, 3, 4]),
        stimuli_start=0,
        stimuli_end=1,
    )

    assert (plr.normalize(0, 1) == np.array([0, 2, 4, 6, 8])).all()


def test_select_subsection():
    plr = PupilLightResponse(
        time=np.array([0, 1, 2, 3, 4]),
        size=np.array([0, 1, 2, 3, 4]),
        stimuli_start=1,
        stimuli_end=2,
    )

    with pytest.raises(ValueError):
        plr.select_subsection(1, 0)
    with pytest.raises(ValueError):
        plr.select_subsection(-1, 0)
    with pytest.raises(ValueError):
        plr.select_subsection(0, 5)
    with pytest.raises(ValueError):
        plr.select_subsection(0, 6)
    with pytest.raises(ValueError):
        plr.select_subsection(5, 6)
    with pytest.raises(ValueError):
        plr.select_subsection(4, 5)

    time, size = plr.select_subsection(0, 1)
    np.testing.assert_allclose(time, np.array([0, 1]))
    np.testing.assert_allclose(size, np.array([0, 1]))

    time, size = plr.select_subsection(1, 2)
    np.testing.assert_allclose(time, np.array([1, 2]))
    np.testing.assert_allclose(size, np.array([1, 2]))

    time, size = plr.select_subsection(2, 3)
    np.testing.assert_allclose(time, np.array([2, 3]))
    np.testing.assert_allclose(size, np.array([2, 3]))

    time, size = plr.select_subsection(3, 4)
    np.testing.assert_allclose(time, np.array([3, 4]))
    np.testing.assert_allclose(size, np.array([3, 4]))

    time, size = plr.select_subsection(0, 3)
    np.testing.assert_allclose(time, np.array([0, 1, 2, 3]))
    np.testing.assert_allclose(size, np.array([0, 1, 2, 3]))
