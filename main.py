import cv2
import numpy as np
from ct.sinogram import radon


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


def main():
    image = cv2.imread("images/Kropka.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sinogram = radon.create_sinogram(gray, max(gray.shape) // 2, 90, 180, 180)
    cv2.imshow("original", gray)
    cv2.imshow("sinogram", sinogram.astype(np.uint8))
    cv2.waitKey()


if __name__ == '__main__':
    main()
