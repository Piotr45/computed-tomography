import cv2
import numpy as np
from ct.sinogram import radon
from ct.sinogram import iradon
from ct.ct import CT


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
    image = cv2.imread("images/Shepp_logan.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ct = CT(gray, 180, 180, 180)
    sinogram, reconstruction = ct.run()
    # cv2.imwrite("original", gray)
    cv2.imwrite("exp/sinogram-180-180-180.png", sinogram)
    cv2.imwrite("exp/reconstructed-180-180-180.png", reconstruction)
    cv2.waitKey()


if __name__ == '__main__':
    main()
