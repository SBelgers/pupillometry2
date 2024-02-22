import numpy as np


class PupilLightResponse:

    def __init__(
        self,
        time: np.ndarray[float],
        size: np.ndarray[float],
        stimuli_start: float = None,
        stimuli_end: float = None,
    ):
        if len(size) != len(time):
            raise ValueError(
                "Pupil size and time lists must be the same length"
            )
        if (stimuli_start is not None) and (stimuli_end is not None):
            if stimuli_start >= stimuli_end:
                raise ValueError("Pulse start must be before pulse end")
            if stimuli_start < time[0] or stimuli_end > time[-1]:
                raise ValueError(
                    "Pulse start and end must be within the time range"
                )

        self.size = size
        self.time = time
        self.stimuli_start = stimuli_start
        self.stimuli_end = stimuli_end

    def __repr__(self) -> str:
        return f"PupilLightResponse(time={self.time}, size={self.size}, stimuli_start={self.stimuli_start}, stimuli_end={self.stimuli_end})"

    def get_size(
        self, start_time: float = None, end_time: float = None
    ) -> np.ndarray[float]:
        if start_time is None:
            start_time = self.time[0]
        if end_time is None:
            end_time = self.time[-1]
        return self.size[(self.time >= start_time) & (self.time <= end_time)]

    def get_time(
        self, start_time: float = None, end_time: float = None
    ) -> np.ndarray[float]:
        if start_time is None:
            start_time = self.time[0]
        if end_time is None:
            end_time = self.time[-1]
        return self.time[(self.time >= start_time) & (self.time <= end_time)]

    def get_start_time(self) -> float:
        return self.get_time()[0]

    def get_end_time(self) -> float:
        print("end time", self.get_time()[-1])
        return self.get_time()[-1]

    def get_stimuli_start(self) -> float:
        return self.stimuli_start

    def get_t_max_constriction(self, stimuli_only: bool = True) -> float:
        # Todo: add testing for nan_policy="omit"
        # TODO add check on get_stimuli_start is None
        if not stimuli_only:
            return self.get_time()[np.nanargmin(self.get_size())]
        else:
            stimuli_start = self.get_stimuli_start()
            if stimuli_start is None:
                raise ValueError("Stimuli start is not defined")
            stimuli_end = self.get_stimuli_end()
            if stimuli_end is None:
                raise ValueError("Stimuli end is not defined")
            time, size = self.select_subsection(stimuli_start, stimuli_end)
            return time[np.nanargmin(size)]

    def get_stimuli_end(self) -> float:
        return self.stimuli_end

    def get_p0(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        return self.select_subsection(
            self.get_start_time(), self.get_stimuli_start()
        )

    def get_p1(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        return self.select_subsection(
            self.get_stimuli_start(), self.get_t_max_constriction()
        )

    def get_p2(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        return self.select_subsection(
            self.get_t_max_constriction() + 1, self.get_stimuli_end()
        )
        # TODO fix the +1, it's a hack to avoid the t_max_constriction point

    def get_p3(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        return self.select_subsection(
            self.get_stimuli_end(), self.get_end_time()
        )

    def get_baseline(
        self, baseline_start: float, baseline_end: float
    ) -> float:
        if baseline_start > baseline_end:
            raise ValueError("Baseline start must be before baseline end")
        if baseline_start < self.time[0] or baseline_end > self.time[-1]:
            raise ValueError(
                "Baseline start and end must be within the time range"
            )
        baseline = 0
        count = 0
        for i, t in enumerate(self.time):
            if baseline_start <= t <= baseline_end:
                baseline += self.get_size()[i]
                count += 1
        return baseline / count

    def normalize(self, baseline_start: float, baseline_end: float):
        baseline = self.get_baseline(baseline_start, baseline_end)
        return self.get_size() / baseline

    def select_subsection(
        self, start_t: float, end_t: float
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        if start_t > end_t:
            raise ValueError("Start must be before end")
        if start_t < self.time[0] or end_t > self.time[-1]:
            raise ValueError("Start and end must be within the time range")
        indices = np.where((self.time >= start_t) & (self.time <= end_t))
        return self.get_time()[indices], self.get_size()[indices]

    def plot(
        self,
        include_t_max_constriction: bool = True,
        include_stimuli: bool = True,
        show: bool = True,
        ax=None,
        **kwargs,
    ):
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        ax.scatter(self.time, self.size, **kwargs)
        if include_t_max_constriction:
            ax.axvline(
                self.get_t_max_constriction(), color="red", linestyle="--"
            )
        if include_stimuli:
            ax.axvline(self.get_stimuli_start(), color="green", linestyle="--")
            ax.axvline(self.get_stimuli_end(), color="green", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pupil size")
        if show:
            plt.show()
        return ax

    # def plot_coefficients(
    #     self,
    #     formula: function,
    #     coefficients: tuple[float],
    #     show=True,
    #     ax=None,
    #     **kwargs,
    # ):
    #     if ax is None:
    #         import matplotlib.pyplot as plt

    #         fig, ax = plt.subplots()
    #     ax.plot(self.time, formula(self.time, *coefficients), **kwargs)
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Pupil size")
    #     return ax
