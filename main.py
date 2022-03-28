import cv2
import numpy as np
from ct.radon import radon
from ct.inverse_radon import iradon
from ct.common import *
from ct.ct import CT
from ct.common import preprocess


def main():
    image = cv2.imread("images/Paski2.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ct = CT(gray, 2, 1, 180, 180)
    sinogram, output = ct.run()
    cv2.imshow("original", image)
    cv2.imshow("reconstructed", output)
    cv2.imshow("sinogram", sinogram)
    cv2.waitKey()
    # print(.data_element("Rows"))
    # print(preprocess.load_example(2).shape)
    # cv2.imshow("dicom", preprocess.load_dicom_file("dicom_files/Paski2.dcm"))
    # cv2.waitKey()


if __name__ == '__main__':
    main()
