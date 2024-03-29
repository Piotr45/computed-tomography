import cv2
import numpy as np
import os
import sys


def load_image(filename: str) -> np.ndarray:
    """
    Loads image and converts it to grayscale.
    :param filename: Path to file
    :return: image in grayscale
    """
    if filename in os.listdir('images'):
        image = cv2.imread(f"{os.getcwd()}\\images\\{filename}", 0)
    else:
        image = cv2.imread(filename, 0)

    if image is None:
        raise FileNotFoundError("File with that name does not exist in 'images' directory or "
                                "you have passed wrong path to file.")
    return image


def save_image(image: np.ndarray, output_file: str) -> None:
    """
    Saves image on disk.
    :param image: image that we want save
    :param output_file: output file name / path
    :return: None
    """
    cv2.imwrite(output_file, image)


def load_example(example: int) -> np.ndarray:
    """
    Loads example file.
    :param example: number from 0 to 8
    :return: image
    """
    EXAMPLES = {
        0: "CT_ScoutView.jpg",
        1: "CT_ScoutView-large.jpg",
        2: "Kolo.jpg",
        3: "Kropka.jpg",
        4: "Kwadraty2.jpg",
        5: "Paski2.jpg",
        6: "SADDLE_PE.JPG",
        7: "SADDLE_PE-large.JPG",
        8: "Shepp_logan.jpg"
    }
    return load_image(f"images/{EXAMPLES[example]}")