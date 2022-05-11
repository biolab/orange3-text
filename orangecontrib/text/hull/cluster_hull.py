from typing import Dict, Tuple, Union

import numpy as np
from shapely import LineString, Point, Polygon

from orangecontrib.text.hull.concave_hull import compute_concave_polygon


def polygon_to_array(polygon: Polygon) -> np.ndarray:
    """Transform Polygon to numpy array of (x, y) pairs"""
    return np.hstack([np.array(x).reshape(-1, 1) for x in polygon.exterior.coords.xy])


def _smoothen_hull(hull: Union[Polygon, LineString, Point]) -> Polygon:
    """Move hull away from the cluster and smoothen it"""
    # first move the hull 0.1 away from the cluster and then move it 0.6 back
    # it results in smooth edges
    return hull.buffer(0.1).buffer(-0.06)


def _xy_range(points: np.array) -> Tuple[float, float, float, float]:
    """Compute min and max value along each axis"""
    x, y = points[:, 0], points[:, 1]
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    # if max and min equal set them for 1 apart - same behaviour in scatter-plot
    if x_max - x_min == 0:
        x_min -= 0.5
        x_max += 0.5
    if y_max - y_min == 0:
        y_min -= 0.5
        y_max += 0.5
    return x_min, x_max, y_min, y_max


def _normalize(points: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Normalize points on the scale between 0 and 1. Widget can have different
    ranges. Computing in original space would cause different visual offset from
    cluster for x and y when ranges not same.
    """
    x, y = points[:, 0:1], points[:, 1:2]
    x_min, x_max, y_min, y_max = _xy_range(points)
    points = np.hstack(((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)))
    return points, x_min, x_max, y_min, y_max


def _denormalize(
    points: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float
) -> np.ndarray:
    """Reverse effect of normalization by _normalize function"""
    xd, yd = (x_max - x_min), (y_max - y_min)
    return np.hstack((points[:, 0:1] * xd + x_min, points[:, 1:2] * yd + y_min))


def compute_hulls(points: np.ndarray, clusters: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Function computes the concave hull around cluster.

    Parameters
    ----------
    points
        Visualisation coordinates - embeddings
    clusters
       Cluster indices for each item.

    Returns
    -------
    The points of the concave hull. Dictionary with cluster index
    as a key and array of points as a value - [[x1, y1], [x2, y2], [x3, y3], ...]
    """
    assert points.shape[1] == 2, "Embedding must have two columns"
    points = points.astype(float)
    points, x_min, x_max, y_min, y_max = _normalize(points)

    hulls = {}
    for cl in set(clusters) - {-1}:
        cpoints = points[clusters == cl]

        # subsample when more than 1000 points - to speed up the algorithm
        if cpoints.shape[0] > 1000:
            cpoints = cpoints[np.random.randint(cpoints.shape[0], size=1000), :]

        hull = compute_concave_polygon(cpoints)
        hull = _smoothen_hull(hull)
        hull = polygon_to_array(hull)
        hulls[cl] = _denormalize(hull, x_min, x_max, y_min, y_max)
    return hulls


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from Orange.data import Table

    table = Table("iris")
    clusters = np.array([1] * 50 + [2] * 50 + [3] * 50)

    hulls_ = compute_hulls(table.X[:, :2], clusters)
    plt.plot(table.X[:, 0], table.X[:, 1], "o")
    for k, h in hulls_.items():
        plt.plot(h[:, 0], h[:, 1])
    plt.show()
