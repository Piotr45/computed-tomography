import numpy as np
from typing import Tuple, List
import skimage.exposure
from ..utils.utils import *


def inverse_radon(sinogram: np.ndarray, radius: int, scans: int, detectors: int,
                  spread: float, animate: bool = False) \
        -> Tuple[np.ndarray, List[np.ndarray]]:
    diameter = 2 * radius

    reconstructed_image = np.zeros((diameter, diameter))
    offset = np.array((radius, radius))

    spread = np.deg2rad(spread)

    results = []

    for (i, (line, rotation)) in enumerate(zip(sinogram, np.linspace(0, 2 * np.pi, scans))):
        emitter = create_offset(calculate_emitter_position(radius, rotation), offset)
        detections = create_offset(calculate_detection_positions(radius, rotation, spread,
                                                                 detectors), offset)

        for (detection, value) in zip(detections.T, line):
            for point in bresenham(emitter, detection):
                reconstructed_image[tuple(point)] += value
        if animate:
            results.append(rescale_array(reconstructed_image, (0, 1)))

    return rescale_array(reconstructed_image).astype(np.uint8), results
