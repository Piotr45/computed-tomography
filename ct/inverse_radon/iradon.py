import numpy as np
from typing import Tuple
from ..common.line_radiation import LineRadiation
from ..common.round import RoundEmitter
import skimage.exposure


def perform_inverse_radon_transform(shape: Tuple[int, int], sinogram: np.ndarray,
                                    rotate_angle: float, theta: float, detector_distance: int,
                                    ) -> np.ndarray:
    """
    Performs inverse radon transform on given sinogram.
    :param shape: shape of original image
    :param sinogram: sinogram of original image
    :param rotate_angle: the angle by which we want to rotate every iteration (in degrees)
    :param theta: theta angle (in degrees)
    :param detector_distance: distance between far-left and far-right detector (in pixels)
    :return: reconstructed image
    """
    diameter = np.sqrt(shape[0] ** 2 + shape[1] ** 2)
    round_emitter = RoundEmitter((shape[0] / 2, shape[1] / 2), diameter / 2,
                                 np.deg2rad(theta), sinogram.shape[1], detector_distance)

    image = np.zeros(shape)
    inverse_radon = LineRadiation(image, LineRadiation.Operation.MEAN)
    rotate_angle_rad = np.deg2rad(rotate_angle)

    for angle in range(sinogram.shape[0]):
        inverse_radon.next(sinogram[angle], round_emitter.emitter, round_emitter.detectors)
        round_emitter.rotate(rotate_angle_rad)

    return skimage.exposure.rescale_intensity(inverse_radon.end().astype(np.uint8))