from __future__ import annotations

import queue
import time
from multiprocessing import Queue
from typing import Dict, Generator, List, Optional, Type, TypeVar

MISSING_DATA_MESSAGE = 'Missing data in pipeline package!'


class BaseData:
    def __init__(self) -> None:
        pass


T = TypeVar('T')


class DataCollection:
    def __init__(
        self,
        data: Dict[type, BaseData] = {},
        timestamp: Optional[float] = None
    ) -> None:
        self.data = data
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = time.time()

    def add(self, data: BaseData) -> DataCollection:
        self.data[type(data)] = data
        return self

    def has(self, data_type: Type) -> bool:
        return data_type in self.data

    def get(self, data_type: Type[T]) -> T:
        return self.data.get(data_type)  # type: ignore

    def is_closed(self) -> bool:
        return CloseData in self.data or ExceptionCloseData in self.data


class CloseData(BaseData):
    def __init__(self) -> None:
        super().__init__()


class ExceptionCloseData(CloseData):
    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self.exception = exception


def pipeline_data_generator(
    input_queue: Queue[DataCollection],
    output_queue: Queue[DataCollection],
    expected_data: List[Type]
) -> Generator[DataCollection, None, None]:
    closing = False
    try:
        while not closing:
            try:
                data = input_queue.get(timeout=0.01)
                if data.is_closed():
                    closing = True
                    output_queue.put(data)
                    break

                assert all(data.has(ed)
                           for ed in expected_data), MISSING_DATA_MESSAGE
                yield data
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        output_queue.put(DataCollection().add(ExceptionCloseData(e)))
