import pytest
from pupillometry.pupilFit import PupilFit
import numpy as np

from pupillometry.pupilFit import (
    p0_formula,
    p1_formula,
    p2_formula,
    p3_formula,
    get_coefficients,
    PupilFit,
)


@pytest.fixture()
def pupil_fit() -> PupilFit:
    return PupilFit(
        start_time=0,
        end_time=40,
        stimuli_start=10,
        stimuli_end=20,
        t_max_constriction=15,
        p0_coefficients=(1, 100),
        p1_coefficients=(-5, 110),
        p2_coefficients=(2, 85),
        p3_coefficients=(3, -0.2, 95),
    )


@pytest.fixture()
def pupil_fit_empty() -> PupilFit:
    return PupilFit()


def test_pre_stimulus_onset_formula():
    x = np.array([1, 2, 3, 4, 5])
    coefficients = [
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    outcomes = [
        np.array([5, 7, 9, 11, 13]),
        np.array([7, 10, 13, 16, 19]),
        np.array([9, 13, 17, 21, 25]),
    ]
    for i, (m, c) in enumerate(coefficients):
        np.testing.assert_allclose(p0_formula(x, m, c), outcomes[i])


def test_stimulus_onset_to_max_constriction_formula():
    x = np.array([1, 2, 3, 4, 5])
    coefficients = [
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    outcomes = [
        np.array([5, 7, 9, 11, 13]),
        np.array([7, 10, 13, 16, 19]),
        np.array([9, 13, 17, 21, 25]),
    ]
    for i, (m, c) in enumerate(coefficients):
        np.testing.assert_allclose(p1_formula(x, m, c), outcomes[i])


def test_max_constriction_to_stimulus_offset_formula():
    x = np.array([1, 2, 3, 4, 5])
    coefficients = [
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    outcomes = [
        np.array([5, 7, 9, 11, 13]),
        np.array([7, 10, 13, 16, 19]),
        np.array([9, 13, 17, 21, 25]),
    ]
    for i, (m, c) in enumerate(coefficients):
        np.testing.assert_allclose(p2_formula(x, m, c), outcomes[i])


def test_post_stimulus_offset_formula():
    x = np.array([1, 2, 3, 4, 5])
    coefficients = [
        (2, 3, 4),
        (3, 4, 5),
    ]
    outcomes = [
        np.array(
            [
                4.41710738e01,
                8.10857587e02,
                1.62101679e04,
                3.25513583e05,
                6.53803874e06,
            ]
        ),
        np.array(
            [
                1.68794450e02,
                8.94787396e03,
                4.88269374e05,
                2.66583366e07,
                1.45549559e09,
            ]
        ),
    ]
    for i, (s, k, p) in enumerate(coefficients):
        np.testing.assert_allclose(p3_formula(x, s, k, p), outcomes[i])


def test_class_creation(
    pupil_fit_empty: PupilFit, pupil_fit: PupilFit
) -> None:
    assert pupil_fit_empty.start_time is None
    assert pupil_fit_empty.end_time is None
    assert pupil_fit_empty.stimuli_start is None
    assert pupil_fit_empty.stimuli_end is None
    assert pupil_fit_empty.t_max_constriction is None
    assert pupil_fit_empty.p0_coefficients is None
    assert pupil_fit_empty.p1_coefficients is None
    assert pupil_fit_empty.p2_coefficients is None
    assert pupil_fit_empty.p3_coefficients is None

    assert pupil_fit.start_time == 0
    assert pupil_fit.end_time == 40
    assert pupil_fit.stimuli_start == 10
    assert pupil_fit.stimuli_end == 20
    assert pupil_fit.t_max_constriction == 15
    assert pupil_fit.p0_coefficients == (1, 100)
    assert pupil_fit.p1_coefficients == (-5, 110)
    assert pupil_fit.p2_coefficients == (2, 85)
    assert pupil_fit.p3_coefficients == (3, -0.2, 95)


def test_get_time(pupil_fit: PupilFit, pupil_fit_empty: PupilFit) -> None:
    np.testing.assert_allclose(
        pupil_fit.get_time(start_time=0, end_time=40, time_step=1),
        np.arange(0, 40, 1),
    )
    np.testing.assert_allclose(
        pupil_fit.get_time(start_time=10, end_time=20, time_step=1),
        np.arange(10, 20, 1),
    )
    np.testing.assert_allclose(
        pupil_fit.get_time(start_time=10, end_time=20, time_step=2),
        np.arange(10, 20, 2),
    )
    np.testing.assert_allclose(
        pupil_fit.get_time(start_time=10, end_time=20, time_step=0.2),
        np.arange(10, 20, 0.2),
    )
    np.testing.assert_allclose(pupil_fit.get_time(), np.arange(0, 40, 1))
    np.testing.assert_allclose(
        pupil_fit.get_time(start_time=10), np.arange(10, 40, 1)
    )
    with pytest.raises(ValueError):
        pupil_fit_empty.get_time()


def test_get_size(pupil_fit: PupilFit, pupil_fit_empty: PupilFit) -> None:
    with pytest.raises(ValueError):
        pupil_fit_empty.get_size()
    with pytest.raises(ValueError):
        pupil_fit_empty.get_size(start_time=10)
    with pytest.raises(ValueError):
        pupil_fit_empty.get_size(end_time=20)
    with pytest.raises(ValueError):
        pupil_fit_empty.get_size(start_time=10, end_time=12)

    test_cases = [
        {
            "start_time": 0,
            "end_time": 10,
            "time_step": 1,
            "expected": np.arange(100, 110, 1),
        },
        {
            "start_time": 10,
            "end_time": 15,
            "time_step": 1,
            "expected": np.arange(60, 35, -5),
        },
        {
            "start_time": 16,
            "end_time": 20,
            "time_step": 1,
            "expected": np.arange(117, 125, 2),
        },
        {
            "start_time": 20,
            "end_time": 40,
            "time_step": 1,
            "expected": np.array(
                [
                    95.05494692,
                    95.04498673,
                    95.03683202,
                    95.03015551,
                    95.02468924,
                    95.02021384,
                    95.01654969,
                    95.01354974,
                    95.01109359,
                    95.00908266,
                    95.00743626,
                    95.00608829,
                    95.00498467,
                    95.0040811,
                    95.00334133,
                    95.00273565,
                    95.00223976,
                    95.00183376,
                    95.00150135,
                    95.0012292,
                ]
            ),
        },
        {
            "start_time": 8,
            "end_time": 12,
            "time_step": 1,
            "expected": np.array([108, 109, 60, 55]),
        },
        {
            "start_time": 13,
            "end_time": 17,
            "time_step": 1,
            "expected": np.array([45, 40, 35, 117]),
        },
        {
            "start_time": 18,
            "end_time": 22,
            "time_step": 1,
            "expected": np.array([121.0, 123.0, 95.05494692, 95.04498673]),
        },
    ]
    for test_case in test_cases:
        np.testing.assert_allclose(
            pupil_fit.get_size(
                start_time=test_case["start_time"],
                end_time=test_case["end_time"],
                time_step=test_case["time_step"],
            ),
            test_case["expected"],
        )

    np.testing.assert_allclose(
        pupil_fit.get_size(start_time=0, end_time=60, time_step=1),
        np.concatenate(
            [
                p0_formula(np.arange(0, 10, 1), *pupil_fit.p0_coefficients),
                p1_formula(np.arange(10, 16, 1), *pupil_fit.p1_coefficients),
                p2_formula(np.arange(16, 20, 1), *pupil_fit.p2_coefficients),
                p3_formula(np.arange(20, 60, 1), *pupil_fit.p3_coefficients),
            ]
        ),
    )


def test_get_t_max_constriction(pupil_fit: PupilFit) -> None:
    assert pupil_fit.get_t_max_constriction() == 15


def test_get_p0(pupil_fit: PupilFit) -> None:
    np.testing.assert_allclose(
        pupil_fit.get_p0(),
        (
            np.arange(0, 10, 1),
            p0_formula(np.arange(0, 10, 1), *pupil_fit.p0_coefficients),
        ),
    )


def test_get_p1(pupil_fit: PupilFit) -> None:
    np.testing.assert_allclose(
        pupil_fit.get_p1(),
        (
            np.arange(10, 15, 1),
            p1_formula(np.arange(10, 15, 1), *pupil_fit.p1_coefficients),
        ),
    )


def test_get_p2(pupil_fit: PupilFit) -> None:
    np.testing.assert_allclose(
        pupil_fit.get_p2(),
        (np.arange(16, 20, 1), np.array([117, 119, 121, 123])),
    )


def test_get_p3(pupil_fit: PupilFit) -> None:
    print(
        pupil_fit.get_p3(),
        (
            np.arange(20, 40, 1),
            p3_formula(np.arange(20, 40, 1), *pupil_fit.p3_coefficients),
        ),
    )
    np.testing.assert_allclose(
        pupil_fit.get_p3(),
        (
            np.arange(20, 40, 1),
            p3_formula(np.arange(20, 40, 1), *pupil_fit.p3_coefficients),
        ),
    )


def test_get_coefficients(pupil_fit: PupilFit) -> None:
    print(pupil_fit.get_p0())
    np.testing.assert_allclose(
        get_coefficients(pupil_fit)[0],
        np.array([1, 100]),
    )
    print(pupil_fit.get_p1())

    np.testing.assert_allclose(
        get_coefficients(pupil_fit)[1],
        np.array([-5, 110]),
    )
    print(pupil_fit.get_p2())

    np.testing.assert_allclose(
        get_coefficients(pupil_fit)[2],
        np.array([2, 85]),
    )
    print(pupil_fit.get_p3())

    np.testing.assert_allclose(
        get_coefficients(pupil_fit)[3],
        np.array([3, -0.2, 95]),
    )
