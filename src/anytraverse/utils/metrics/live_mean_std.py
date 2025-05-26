from typing import Tuple


class LiveMeanStd:
    _mean: float
    _std: float
    _n: int
    _latest: float

    def __init__(self) -> None:
        self._mean = 0
        self._std = 0
        self._n = 0
        self._latest = 0

    def update(self, value) -> None:
        self._n += 1
        self._latest = value
        self._mean += (value - self._mean) / self._n
        self._std += ((value - self._mean) ** 2 - self._std) / self._n

    @property
    def latest(self) -> float:
        return self._latest

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    @property
    def values(self) -> Tuple[float, float]:
        return self._mean, self._std

    def reset(self) -> None:
        self._mean = 0
        self._std = 0
        self._latest = 0
        self._n = 0

    def to_plus_minus(self, n=2) -> str:
        return f"{self._mean:.{n}f} ± {self._std:.{n}f}"

    def __str__(self):
        return f"LiveMean(mean={self._mean}, std={self._std}, n={self._n})"

    def __repr__(self):
        return str(self)
