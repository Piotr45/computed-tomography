import numpy as np
from typing import Tuple, List, Callable
from ..utils.utils import *
from ..utils.bresenham import bresenham
from skimage.draw import line


def create_sinogram(image: np.ndarray, radius: int, scans: int, detectors: int, spread: int,
                    animate: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
    spread = np.deg2rad(spread)

    sinogram = []
    offset = np.array((radius, radius), dtype=float)

    results = []

    for (i, rotation) in enumerate(np.linspace(0, 2 * np.pi, scans)):
        emitter = create_offset(calculate_emitter_position(radius, rotation), offset)
        detections = create_offset(calculate_detection_positions(radius, rotation, spread,
                                                                 detectors), offset)
        sinogram.append([calculate_point(emitter, detection, image) for detection in detections.T])
        if animate:
            results.append(rescale_array(np.array(sinogram), (0, 1)))

    sinogram = np.array(sinogram)

    return sinogram, results
