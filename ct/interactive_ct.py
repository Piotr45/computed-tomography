import numpy as np
from typing import Tuple
from ct.ct import CT


class InteractiveCT(CT):
    """
    Simulates computer tomography via image reconstruction in interactive mode.
    """
    def __init__(self, image:np.ndarray, rotate_angle: int, start_angle: int,
                 number_of_detectors: int, detector_distance: int):
        """
        :param image: image that we want to be reconstructed
        :param rotate_angle: the angle by which we want to rotate every iteration (in degrees)
        :param start_angle: angle on the circle from which we will start the measurement
        :param number_of_detectors: number of detectors in cluster
        :param detector_distance: distance between far-left and far-right detector (in pixels)
        """
        super().__init__(image, rotate_angle, start_angle, number_of_detectors, detector_distance)

        self.iter = 0
        self.stop_iteration = 180 // rotate_angle
        self.step = 1
        self.rotate_angle = rotate_angle
        self.sinograms = list()
        self.results = list()

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        return super().run()

    def get_images(self) -> Tuple[list, list]:
        """

        :return:
        """
        return self.sinograms, self.results

    def reset_iteration(self) -> None:
        self.iter = 0

    def save_radon_frame(self, sinogram: np.ndarray) -> None:
        if not self.iter % self.step and self.iter < self.stop_iteration:
            sinogram = np.array(sinogram.T, copy=True)
            self.sinograms.append(sinogram)
        self.iter += 1

    def save_inverse_radon_frame(self, image:np.ndarray, pixel_lines: np.ndarray) -> None:
        if not self.iter % self.step and self.iter < self.stop_iteration:
            iradon_image = np.array(image, copy=True)
            scaler = iradon_image / pixel_lines
            scaler = np.max(scaler)
            iradon_image /= scaler if scaler > 0 else 1
            self.results.append(iradon_image)
        self.iter += 1
