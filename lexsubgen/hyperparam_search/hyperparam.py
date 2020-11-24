from typing import List, Optional

import numpy as np


class Hyperparam:
    def __init__(
        self,
        values: List = (),
        name: Optional[str] = None,
        need_to_restart: bool = True,
    ):
        self._values = values
        self.name = name
        self.need_to_restart = need_to_restart

    @property
    def values(self):
        return self._values


class LinspaceHyperparam(Hyperparam):
    def __init__(
        self,
        start: float,
        end: float,
        size: int = 10,
        name: Optional[str] = None,
        need_to_restart: bool = True,
    ):
        super(LinspaceHyperparam, self).__init__(
            name=name, need_to_restart=need_to_restart
        )
        assert size > 0
        self.start = start
        self.end = end
        self.size = size

    @property
    def values(self):
        return np.linspace(start=self.start, stop=self.end, num=self.size)


class LogspaceHyperparam(Hyperparam):
    def __init__(
        self,
        start: float,
        end: float,
        size: int = 10,
        base: float = 10.0,
        name: Optional[str] = None,
        need_to_restart: bool = True,
    ):
        super(LogspaceHyperparam, self).__init__(
            name=name, need_to_restart=need_to_restart
        )
        assert size > 0
        self.start = start
        self.end = end
        self.size = size
        self.base = base

    @property
    def values(self):
        return np.logspace(
            start=self.start, stop=self.end, num=self.size, base=self.base
        )
