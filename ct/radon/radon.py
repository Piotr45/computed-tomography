import numpy as np
from typing import Tuple, List
from ..common.round import RoundEmitter
from skimage.draw import line


def calculate_radiation(image: np.ndarray, number_of_detectors: int,
                        emitter_position: Tuple[int, int],
                        detectors_positions: List[Tuple[int, int]]) -> np.ndarray:
    """
    Calculates radiation from emitter to detectors by averaging pixel values on the emitter-detector
    line for every detector in cluster.
    :param image: image that we want to reconstruct passed as numpy ndarray
    :param number_of_detectors: number of detectors in cluster
    :param emitter_position: position of emitter
    :param detectors_positions: list of detector positions
    :return: radiation
    """
    image_width, image_height = image.shape
    detectors_brightness = np.ndarray(shape=(number_of_detectors,))

    for i, detector in enumerate(detectors_positions):
        rr, cc = line(emitter_position[1], emitter_position[0], detector[1], detector[0])
        start = 0
        for j in range(len(rr)):
            if 0 <= rr[j] < image_width and 0 <= cc[j] < image_height:
                start = j
                break

        stop = 0
        for j in range(len(rr) - 1, start + 1, -1):
            if 0 <= rr[j] < image_width and 0 <= cc[j] < image_height:
                stop = j
                break

        pixel_brightness = np.sum(image[rr[start:stop + 1], cc[start:stop + 1]])
        detectors_brightness[i] = pixel_brightness / (stop - start + 1)

    return detectors_brightness


def generate_sinogram(image: np.ndarray, theta: int, rotate_angle: int, number_of_detectors: int,
                      detector_distance: int) -> np.ndarray:
    """
    This function constructs sinogram from given image using Radon Transform.
    :param theta: theta angle (in degrees)
    :param image: image that we want to reconstruct passed as numpy ndarray
    :param rotate_angle: the angle by which we want to rotate every iteration (in degrees)
    :param number_of_detectors: number of detectors in cluster
    :param detector_distance: distance between far-left and far-right detector (in pixels)
    :return: sinogram of an image
    """

    diameter = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
    round_emitter = RoundEmitter((image.shape[0] / 2, image.shape[1] / 2), diameter / 2,
                                 np.deg2rad(theta), number_of_detectors, detector_distance)

    rotate_angle_rad = np.deg2rad(theta)
    sinogram = np.zeros(shape=(180 // rotate_angle, number_of_detectors))

    for i in range(180 // rotate_angle):
        sinogram[i] = calculate_radiation(image, number_of_detectors, round_emitter.emitter,
                                          round_emitter.detectors)
        round_emitter.rotate(rotate_angle_rad)

    return sinogram
