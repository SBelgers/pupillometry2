import numpy as np
from pupillometry.pupilLightResponse import PupilLightResponse

from scipy.optimize import curve_fit


def check_overlap(start1, end1, start2, end2):
    if start1 < end2 and start2 < end1:
        return True
    else:
        return False


def get_coefficients(
    pupilLightResponse: PupilLightResponse,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    time, size = pupilLightResponse.get_p0()
    p0_coefficients, _ = curve_fit(f=p0_formula, xdata=time, ydata=size)
    time, size = pupilLightResponse.get_p1()
    p1_coefficients, _ = curve_fit(p1_formula, time, size)
    time, size = pupilLightResponse.get_p2()
    p2_coefficients, _ = curve_fit(p2_formula, time, size)
    time, size = pupilLightResponse.get_p3()
    p3_coefficients, _ = curve_fit(p3_formula, time, size)
    return p0_coefficients, p1_coefficients, p2_coefficients, p3_coefficients


def p0_formula(x, m, c):
    return m * x + c


def p1_formula(x, m, c):
    return m * x + c


def p2_formula(x, m, c):
    return m * x + c


def p3_formula(x, s, k, p):
    return s * np.exp(k * (x)) + p


class PupilFit(PupilLightResponse):
    def __init__(
        self,
        start_time: float = None,
        end_time: float = None,
        stimuli_start: float = None,
        stimuli_end: float = None,
        t_max_constriction: float = None,
        p0_coefficients: tuple[float, float] = None,
        p1_coefficients: tuple[float, float] = None,
        p2_coefficients: tuple[float, float] = None,
        p3_coefficients: tuple[float, float, float] = None,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.stimuli_start = stimuli_start
        self.stimuli_end = stimuli_end
        self.t_max_constriction = t_max_constriction
        self.p0_coefficients = p0_coefficients
        self.p1_coefficients = p1_coefficients
        self.p2_coefficients = p2_coefficients
        self.p3_coefficients = p3_coefficients

    def __repr__(self) -> str:
        return f"PupilFit(stimuli_start={self.stimuli_start}, stimuli_end={self.stimuli_end}, t_max_constriction={self.t_max_constriction}, p0_coefficients={self.p0_coefficients}, p1_coefficients={self.p1_coefficients}, p2_coefficients={self.p2_coefficients}, p3_coefficients={self.p3_coefficients})"

    def get_time(
        self,
        *,
        time_step: float = 1,
        start_time: float = None,
        end_time: float = None,
    ) -> np.ndarray[float]:
        if start_time is None:
            if self.start_time is None:
                raise ValueError(
                    "start time not provided, nor is it set in the object"
                )
            start_time = self.start_time
        if end_time is None:
            if self.end_time is None:
                raise ValueError(
                    "end time not provided, nor is it set in the object"
                )
            end_time = self.end_time
        return np.arange(start_time, end_time, time_step, dtype=float)

    def get_size(
        self,
        *,
        time_step: float = 1,
        start_time: float = None,
        end_time: float = None,
    ) -> tuple[np.ndarray[float]]:
        if start_time is None:
            if self.start_time is None:
                raise ValueError(
                    "start time not provided, nor is it set in the object"
                )
            start_time = self.start_time
        if end_time is None:
            if self.end_time is None:
                raise ValueError(
                    "end time not provided, nor is it set in the object"
                )
            end_time = self.end_time
        stimuli_start = self.stimuli_start
        stimuli_end = self.stimuli_end
        if stimuli_start is None:
            raise ValueError("stimuli start is None")
        if stimuli_end is None:
            raise ValueError("stimuli end is None")
        t_max_constriction = self.t_max_constriction
        if t_max_constriction is None:
            raise ValueError("t_max_constriction is None")

        p0_time = np.arange(
            start_time, min(stimuli_start, end_time), time_step, dtype=float
        )
        p1_time = np.arange(
            max(stimuli_start, start_time),
            min(t_max_constriction + time_step, end_time),
            time_step,
            dtype=float,
        )
        p2_time = np.arange(
            max(start_time, t_max_constriction + time_step),
            min(stimuli_end, end_time),
            time_step,
            dtype=float,
        )
        p3_time = np.arange(
            max(stimuli_end, start_time), end_time, time_step, dtype=float
        )

        p0_size = p0_formula(p0_time, *self.p0_coefficients)
        p1_size = p1_formula(p1_time, *self.p1_coefficients)
        p2_size = p2_formula(p2_time, *self.p2_coefficients)
        p3_size = p3_formula(p3_time, *self.p3_coefficients)
        size = np.concatenate(
            [
                np.atleast_1d(p0_size),
                np.atleast_1d(p1_size),
                np.atleast_1d(p2_size),
                np.atleast_1d(p3_size),
            ]
        )
        return size

    def select_subsection(
        self,
        start_time: float = None,
        end_time: float = None,
        *,
        time_step: float = 1,
    ) -> np.ndarray[float]:
        return (
            self.get_time(
                time_step=time_step, start_time=start_time, end_time=end_time
            ),
            self.get_size(
                time_step=time_step, start_time=start_time, end_time=end_time
            ),
        )

    def get_p3(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        return self.select_subsection(
            self.get_stimuli_end(), self.get_end_time() + 1
        )
        # TODO VERY HACKY SOLUTION, FIX IT.
        # I added 1 to the end time to make sure the end time is
        # included in the range, but this breaks when you use a different
        # time step
