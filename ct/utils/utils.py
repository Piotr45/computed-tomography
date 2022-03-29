import numpy as np
from .bresenham import bresenham


def calculate_point(emitter: np.ndarray, detection: np.ndarray, image: np.ndarray) -> int:
    return round(image[tuple(bresenham(emitter, detection).T)].mean())


def create_offset(array: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (array.T + offset).T


def calculate_emitter_position(radius: float, rotation: float) -> np.ndarray:
    return np.clip(radius * np.array([np.cos(rotation), np.sin(rotation)], dtype=float),
                   -radius + 1, radius - 1)


def calculate_detection_positions(radius: float, rotation: float, spread: float, detectors: int) \
        -> np.ndarray:
    angles = rotation + np.pi - spread / 2 + np.arange(detectors) * spread / (detectors - 1)
    return np.clip(radius * np.array([np.cos(angles), np.sin(angles)], dtype=float),
                   -radius + 1, radius - 1)


def create_kernel(size: int) -> np.ndarray:
    h0 = 1
    h = [np.divide(-4, np.square(np.pi * k)) if np.mod(k, 2) else 0 for k in range(size // 2 - 1)]
    return np.array([*h[::-1], h0, *h])


def filter_sinogram(sinogram: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    sinogram = sinogram.copy()
    for i in range(sinogram.shape[0]):
        sinogram[i, :] = np.convolve(sinogram[i, :], kernel, mode='same')
    return sinogram


def rescale_array(arr: np.ndarray, feature=(0, 255)) -> np.ndarray:
    ranges = (arr.min(initial=None), arr.max(initial=None))
    return np.interp(arr, ranges, feature)
