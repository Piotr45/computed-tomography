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
    image = cv2.imread("images/Kolo.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # sinogram = radon.create_sinogram(gray, max(gray.shape) // 2, 90, 180, 120)
    # sinogram = rescale_array(sinogram, (0, 1))
    # sinogram = filter_sinogram(sinogram, create_kernel(21))
    # out = iradon.inverse_radon(sinogram, gray, max(gray.shape) // 2, 90, 180, 120)
    # cv2.imshow("original", gray)
    # cv2.imshow("sinogram", sinogram.astype(np.uint8))
    # cv2.imshow("reconstructed", out)
    # cv2.waitKey()
    ct = CT(gray, 90, 180, 120, True)
    sinogram, reconstruction = ct.run()
    print(len(ct.get_frames()[0]), len(ct.get_frames()[1]))
    cv2.imshow("original", gray)
    cv2.imshow("sinogram", sinogram.astype(np.uint8))
    cv2.imshow("reconstructed", reconstruction)
    cv2.waitKey()


if __name__ == '__main__':
    main()
