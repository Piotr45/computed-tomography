import numpy as np
from typing import Tuple, List, Callable, Union
from skimage.draw import line
from enum import Enum


class LineRadiation:
    """
    Reconstructs original image using sinogram.
    """
    class Operation(Enum):
        MEAN = 1
        SQRT = 2

    def __init__(self, image: np.ndarray, operation_type: Operation):
        self.image = image
        self.amount = np.ones(self.image.shape)

        if operation_type == LineRadiation.Operation.MEAN:
            self.operation = self.next_mean
            self.end_operation = self.end_mean
        elif operation_type == LineRadiation.Operation.SQRT:
            self.operation = self.next_sqrt
            self.end_operation = self.end_sqrt

    def next(self, detectors_values: np.ndarray, emitter_position: Tuple[int, int],
             detectors_positions: List[Tuple[int, int]]) -> None:
        self.find_lines_pixels(detectors_values, emitter_position, detectors_positions,
                               self.operation)

    def end(self) -> np.ndarray:
        self.end_operation()
        return self.image

    def find_lines_pixels(self, detectors_values: np.ndarray, emitter_position: Tuple[int, int],
                          detectors_positions: List[Tuple[int, int]],
                          operation: Callable[[List[int], List[int], Union[int, float]], None]) -> None:
        image_width, image_height = self.image.shape

        for detector_index, detector in enumerate(detectors_positions):
            rr, cc = line(emitter_position[1], emitter_position[0], detector[1], detector[0])

            start = 0
            for i in range(len(rr)):
                if 0 <= rr[i] < image_width and 0 <= cc[i] < image_height:
                    start = i
                    break

            stop = 0
            for i in range(len(rr) - 1, start + 1, -1):
                if 0 <= rr[i] < image_width and 0 <= cc[i] < image_height:
                    stop = i
                    break

            operation(rr[start:stop + 1], cc[start:stop + 1], detectors_values[detector_index])

    def next_mean(self, pixels_x_axis: List[int], pixels_y_axis: List[int],
                  detector_value: Union[int, float]) -> None:
        """
        Adds value of emitter-detector line to the intersected pixels and calculates average of it.
        :param pixels_x_axis: position of pixels in x-axis for corresponding y position
        :param pixels_y_axis: position of pixels in y-axis for corresponding x position
        :param detector_value: detector value vector
        :return: None
        """
        self.image[pixels_x_axis, pixels_y_axis] += detector_value
        self.amount[pixels_x_axis, pixels_y_axis] += 1

    def end_mean(self) -> None:
        x = self.image / self.amount
        x = np.max(x)
        self.image /= x if x > 0 else 1

    def next_sqrt(self, pixels_x_axis: List[int], pixels_y_axis: List[int],
                  detector_value: Union[int, float]) -> None:
        """
        Adds value of emitter-detector line to the intersected pixels and calculates
        square root of it.
        :param pixels_x_axis: position of pixels in x-axis for corresponding y position
        :param pixels_y_axis: position of pixels in y-axis for corresponding x position
        :param detector_value: detector value vector
        :return: None
        """
        self.image[pixels_x_axis, pixels_y_axis] += detector_value

    def end_sqrt(self) -> None:
        self.image = np.sqrt(self.image)
