from typing import Tuple
import numpy as np


class RoundEmitter:
    """
    Simulates rounding emitter and detectors around the image.
    """
    def __init__(self, center: Tuple[float, float], radius: float, start_angle: float,
                 number_of_detectors: int, detector_distance: int):
        """
        :param center: image center position
        :param radius: radius of rounding emitter
        :param start_angle: initial angle (in degrees)
        :param number_of_detectors: number of detectors in cluster
        :param detector_distance: distance between far-left and far-right detector (in pixels)
        """
        self.center = center
        self.radius = radius
        self.theta = start_angle
        self.angle = 0

        self.number_of_detectors = number_of_detectors
        self.detector_distance = detector_distance

        self.emitter = None
        self.detectors = [None for _ in range(self.number_of_detectors)]

        if self.detector_distance > 2 * radius:
            raise ArithmeticError(f"Distance between detector position has to be less or equal to"
                                  f" {int(2 * radius)}!")
        else:
            self.fi = 2 * np.arcsin(self.detector_distance / (2 * self.radius))

        self.detectors_angles = [
            self.theta + np.pi - self.fi / 2 + i * self.fi / (self.number_of_detectors - 1)
            for i in range(self.number_of_detectors)]

        self.calculate_emitter_position()
        self.calculate_detectors_positions()

    def calculate_emitter_position(self) -> None:
        """
        Calculates emitter position.
        :return: None
        """
        new_angle = self.theta + self.angle
        x = self.center[0] + self.radius * np.cos(new_angle)
        y = self.center[1] - self.radius * np.sin(new_angle)
        self.emitter = int(np.round(x)), int(np.round(y))

    def calculate_detectors_positions(self) -> None:
        """
        Calculates position of every detector.
        :return: None
        """
        for i in range(self.number_of_detectors):
            new_angle = self.angle + self.detectors_angles[i]
            x = self.center[0] + self.radius * np.cos(new_angle)
            y = self.center[1] - self.radius * np.sin(new_angle)
            self.detectors[i] = (int(np.round(x)), int(np.round(y)))

    def rotate(self, angle: float) -> None:
        """
        Rotates rounding emitter by certain angle.
        :param angle: angle in radians
        :return: None
        """
        self.angle += angle
        self.calculate_emitter_position()
        self.calculate_detectors_positions()
