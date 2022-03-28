import cv2
import numpy as np
from abc import abstractmethod
from ct.radon import radon
from ct.inverse_radon import iradon
from typing import Tuple


class CT:
    """
    Simulates computer tomography via image reconstruction.
    """
    def __init__(self, image: np.ndarray, rotate_angle: int, start_angle: int,
                 number_of_detectors: int, detector_distance: int):
        """
        :param image: image that we want to be reconstructed
        :param rotate_angle: the angle by which we want to rotate every iteration (in degrees)
        :param start_angle: angle on the circle from which we will start the measurement
        :param number_of_detectors: number of detectors in cluster
        :param detector_distance: distance between far-left and far-right detector (in pixels)
        """
        if rotate_angle <= 0 or rotate_angle >= 180:
            raise ArithmeticError("Rotate angle have to be in range (0, 180)!")
        else:
            self.image = image
            self.rotate_angle = rotate_angle
            self.theta = start_angle
            self.number_of_detectors = number_of_detectors
            self.detector_distance = detector_distance

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs computed tomography.
        :return: sinogram and reconstructed image
        """
        sinogram = radon.generate_sinogram(self.image, self.theta, self.rotate_angle,
                                           self.number_of_detectors, self.detector_distance)

        self.reset_iteration()

        image = iradon.perform_inverse_radon_transform(self.image.shape, sinogram, self.rotate_angle,
                                                       self.theta, self.detector_distance)
        sinogram /= np.max(sinogram)
        return sinogram.T, image

    @abstractmethod
    def save_radon_frame(self, sinogram: np.ndarray) -> None:
        pass

    @abstractmethod
    def save_inverse_radon_frame(self, image:np.ndarray, pixel_lines: np.ndarray) -> None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass
