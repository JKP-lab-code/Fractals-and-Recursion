import numpy as np


def koch_curve(p1, p2, depth):
    """Recursively generate (x, y) vertices along a single Koch curve segment."""
    if depth == 0:
        return [np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)]

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    vec = (p2 - p1) / 3.0

    a = p1 + vec
    b = p1 + 2.0 * vec

    angle = np.deg2rad(60.0)
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)],
    ])
    peak = a + rot @ vec

    return (
        koch_curve(p1, a, depth - 1)[:-1]
        + koch_curve(a, peak, depth - 1)[:-1]
        + koch_curve(peak, b, depth - 1)[:-1]
        + koch_curve(b, p2, depth - 1)
    )


def koch_snowflake(depth):
    """Return a closed (N, 2) NumPy array of vertices for a Koch snowflake."""
    A = np.array([0.0, 0.0])
    B = np.array([0.5, np.sqrt(3.0) / 2.0])
    C = np.array([1.0, 0.0])
    corners = [A, B, C, A]

    points = []
    for start, end in zip(corners, corners[1:]):
        side = koch_curve(start, end, depth)
        points.extend(side[:-1])
    points.append(corners[0])
    return np.array(points)


def sierpinski_triangles(depth):
    """Recursively generate a list of (3, 2) triangle vertex arrays for Sierpiński."""
    def _subdivide(p1, p2, p3, d):
        if d == 0:
            return [np.array([p1, p2, p3], dtype=float)]
        m12 = (p1 + p2) / 2.0
        m23 = (p2 + p3) / 2.0
        m31 = (p3 + p1) / 2.0
        return (
            _subdivide(p1, m12, m31, d - 1)
            + _subdivide(m12, p2, m23, d - 1)
            + _subdivide(m31, m23, p3, d - 1)
        )

    A = np.array([0.0, 0.0])
    B = np.array([0.5, np.sqrt(3.0) / 2.0])
    C = np.array([1.0, 0.0])
    return _subdivide(A, B, C, depth)


def polygon_perimeter(points):
    """Calculate total perimeter (units) of a closed vertex array using NumPy."""
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def polygon_area(points):
    """Calculate enclosed area (square units) using the vectorized Shoelace formula."""
    x = points[:-1, 0]
    y = points[:-1, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    return float(0.5 * np.abs(np.dot(x, y_next) - np.dot(y, x_next)))

