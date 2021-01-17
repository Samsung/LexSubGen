from typing import Sized

import numpy as np


class BatchReader:
    def __init__(self, *data: Sized, batch_size: int = 32):
        """
        Class that handles batch reading of data. It's an iterator
        that subsequently reads data

        Args:
            *data: sequence of data parts that should be read.
                The sizes of all data parts must be the same.
            batch_size: the size of the batch
        """
        self.data = data
        data_parts_num = len(data)
        assert data_parts_num > 0
        self.data_size = len(data[0])
        assert data_parts_num < 1 or all(
            len(datum) == self.data_size for datum in data
        ), f"{[len(datum) for datum in data]}"
        self.batch_size = batch_size
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.data_size:
            cur_idx = self.idx
            self.idx += self.batch_size
            batch = []
            for datum in self.data:
                batch.append(datum[cur_idx : cur_idx + self.batch_size])
            return batch
        else:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        return np.math.ceil(self.data_size / self.batch_size)
