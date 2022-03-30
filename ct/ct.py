import numpy as np
from abc import abstractmethod
from ct.sinogram import radon, iradon
from ct.utils.utils import *
from typing import Tuple, List


class CT:
    """
    Simulates computer tomography via image reconstruction.
    """
    def __init__(self, image: np.ndarray, scans: int, detectors: int, spread: int,
                 animate: bool = False):
        if scans <= 0 or scans >= 180:
            raise ArithmeticError("Scans have to be in range (0, 180)!")
        else:
            self.image = image
            self.scans = scans
            self.detectors = detectors
            self.spread = spread
            self.animate = animate
            self.sinograms = []
            self.reconstructions = []

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs computed tomography.
        :return: sinogram and reconstructed image
        """
        radius = max(self.image.shape) // 2

        sinogram, sinograms = radon.create_sinogram(self.image, radius, self.scans, self.detectors,
                                                    self.spread, self.animate)

        sinogram = rescale_array(sinogram, (0, 1))
        sinogram = filter_sinogram(sinogram, create_kernel(21))
        self.sinograms = sinograms

        reconstruction, reconstructions = iradon.inverse_radon(sinogram, radius, self.scans,
                                                               self.detectors, self.spread,
                                                               self.animate)

        self.reconstructions = reconstructions
        return sinogram, reconstruction

    def get_frames(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.sinograms, self.reconstructions
