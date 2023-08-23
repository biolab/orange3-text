import itertools
from typing import Optional, Union

import numpy as np
from scipy.spatial.distance import cdist
from shapely import LineString, Point, Polygon

from orangecontrib.text.hull.alphashape import alphashape, optimizealpha


HullType = Union[Polygon, Point, LineString]


def _collinear(p0, p1, p2):
    """Test if three points are collinear"""
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-12


def all_collinear(points: np.ndarray) -> bool:
    """
    Test if all points are collinear

    Parameters
    ----------
    points
        The array of points

    Returns
    -------
    True if all points are collinear, else False
    """
    # it is expensive to compare all combinations but in reality we will find
    # nonlinear case fast and finish the loop, the probability when having
    # thousands of points, and they are collinear is low
    for x, y, z in itertools.combinations(points, 3):
        if not _collinear(x, y, z):
            return False
    return True


def _handle_special_cases(points: np.ndarray) -> Optional[HullType]:
    """Handle special cases that cannot be addressed with alphashape module"""
    if len(points) == 1:
        # only one point in dataset - return it as a Point instance
        return Point(points[0])
    if all_collinear(points):
        # delaunay has issues with collinear points - find points them
        # further away return them as a polygon between those two points
        dist = cdist(points, points, metric="euclidean")
        most_distant_idx = np.unravel_index(dist.argmax(), dist.shape)
        most_distant = points[most_distant_idx, :]
        return LineString(most_distant)


def compute_concave_polygon(points: np.ndarray) -> HullType:
    """
    Compute concave hull around points with alphashape module. First find the
    biggest alpha value than still preserve the shape as a one polygon and the
    use alpha to find Polygon that represents the concave hull.

    Parameters
    ----------
    points
        Points to find the convex hull around - array of xy pairs

    Returns
    -------
    Concave hull around points as a polygon
    """
    sp_case = _handle_special_cases(points)
    if sp_case is not None:
        return sp_case

    # limiting upper alpha value speed up bisection - since we don't need the
    # most accurate concave hull we limit upper alpha value such that radius
    # of the circle is 1/10 of the cluster's size
    # it would be the most correct to consider distance between the two most
    # distant points but since computing it is more expensive max of width and
    # height is good enough
    x_diff = points[:, 0].max() - points[:, 0].min()
    y_diff = points[:, 1].max() - points[:, 1].min()
    max_diff = max(x_diff, y_diff)
    upper_limit = 1 / (max_diff / 10)  # alpha = 1 / radius

    # find the biggest alpha that still preserve single polygon
    alpha = optimizealpha(points, max_iterations=50, upper=upper_limit)
    # get the concave hull as a polygon for alpha
    return alphashape(points, alpha)


if __name__ == "__main__":
    import time

    from matplotlib import pyplot as plt

    # fmt: off
    points_ = np.array(
        [
            [10, 9], [9, 18], [16, 13], [11, 15], [12, 14], [18, 12], [2, 14],
            [6, 18], [9, 9], [10, 8], [6, 17], [5, 3], [13, 19], [3, 18],
            [8, 17], [9, 7], [3, 0], [13, 18], [15, 4], [13, 16],
        ]
    )
    # points_ = np.vstack([points_ + (i * 50) for i in range(5)])
    # points_ = np.array(
    #     [
    #         [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [6, 2], [6, 3], [5, 3],
    #         [4, 3], [3, 3], [3, 4], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7], [4, 7],
    #         [3, 7], [3, 8], [3, 9], [4, 9], [5, 9], [6, 9], [6, 10], [6, 11], [5, 11],
    #         [4, 11], [3, 11], [2, 11], [1, 11], [1, 10], [1, 9], [1, 8], [1, 7],
    #         [1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [5, 2], [4, 2], [3, 2], [2, 2],
    #         [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [3, 10],
    #         [4, 10], [5, 10], [3, 6], [4, 6], [5, 6], [4.5, 7], [3, 8.5],
    #     ]
    # )
    # fmt: on

    t = time.time()
    hull_ = compute_concave_polygon(points_)
    hull_x, hull_y = hull_.exterior.coords.xy
    print("Hull computed in", time.time() - t, "s")
    plt.plot(points_[:, 0], points_[:, 1], "o")
    plt.plot(hull_x, hull_y)
    plt.show()
