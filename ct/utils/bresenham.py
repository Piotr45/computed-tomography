import numpy as np


def bresenham(start: np.ndarray, target: np.ndarray) -> np.ndarray:
    (x0, y0) = map(round, start)
    (x1, y1) = map(round, target)
    (dx, sx) = (abs(x1 - x0), 1 if x0 < x1 else -1)
    (dy, sy) = (-abs(y1 - y0), 1 if y0 < y1 else -1)
    error = dx + dy

    points = []
    while True:
        points.append([x0, y0])
        if x0 == x1 and y0 == y1:
            break
        doubled_err = error << 1
        if doubled_err >= dy:
            if x0 == x1:
                break
            error += dy
            x0 += sx
        if doubled_err <= dx:
            if y0 == y1:
                break
            error += dx
            y0 += sy
    return np.array(points, dtype=int)
