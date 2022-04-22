from bisect import bisect_left, bisect_right
from math import sqrt
from typing import Optional, Dict, Iterable, Tuple, List

import numpy as np
import pyclipper
from Orange.data import ContinuousVariable, Domain, Table
from scipy.spatial import Delaunay


def _angle(
    v1: Tuple[np.ndarray, np.ndarray], v2: Tuple[np.ndarray, np.ndarray]
) -> float:
    """
    Compute clockwise angles between v1 and v2. Both vectors are
    given as a tuple with two points ([x1, y1], [x2, y2]) such that
    [x2, y2] of v1 == [x1, y1] of v2.
    The angle is given in degrees between 0 and 2 * pi
    """
    v1, v2 = np.array(v1), np.array(v2)
    x1, y1 = v1[0] - v1[1]
    x2, y2 = v2[1] - v2[0]
    dot = np.sum(x1 * x2 + y1 * y2)  # dot product
    det = x1 * y2 - y1 * x1  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return -angle if angle <= 0 else np.pi + np.pi - angle


def _find_hull(
    edges_list: List, points_list: np.ndarray, starting_edge: int
) -> Tuple[np.ndarray, List]:
    """
    This function return a single hull which starts and ends in starting_edge.

    Parameters
    ----------
    edges_list
        List of edges. Each edge is presented as a tuple of two indices which
        tell the starting and ending note. The index correspond to location
        of point in points_list. This list must be sorted in the ascending order.
    points_list
        Points location. Each point has x and y location.
    starting_edge
        The index of the list where hull starts.

    Returns
    -------
    polygon
        The array with the hull/polygon points
    used_points
        List of booleans that indicates whether each point was used in polygon
    """
    firsts = [x[0] for x in edges_list]  # first elements for bisection
    used_edges = [False] * len(edges_list)
    used_edges[starting_edge] = True

    # remember start and next point
    start, next_point = edges_list[starting_edge]

    # remember current polygon around points
    poly = [points_list[start], points_list[next_point]]

    # we count number of steps to stop iteration in case it is locked
    # in some dead cycle. It can be a result of some unexpected cases.
    count = 0
    while start != next_point and count < len(edges_list):
        # find the index where the first value equal to next_point appear
        ind_left = bisect_left(firsts, next_point)
        # find the index next to the last value
        ind_right = bisect_right(firsts, next_point)

        # check if there are more edges available from the same point
        if ind_right - ind_left > 1:
            # select the most distant one in clockwise direction. It is probably
            # the point on the outer hull - we prevent a hull to discover cycles
            # inside a polygon
            ang = -1
            for i in range(ind_left, ind_right):
                cur_ang = _angle(
                    (poly[-2], poly[-1]), (poly[-1], points_list[edges_list[i][1]])
                )
                if cur_ang > ang:
                    ang = cur_ang
                    ind_left = i
        # save a next point of the polygon
        used_edges[ind_left] = True
        next_point = edges_list[ind_left][1]
        poly.append(points_list[next_point])
        count += 1
    return np.array(poly), used_edges


def _edges_to_polygon(edges: Iterable, points: np.ndarray) -> np.ndarray:
    """
    This function connects edges in polygons. It computes all possible hulls -
    some clusters have more of them when they have a hole in the middle, and
    selects outer hull.

    Parameters
    ----------
    edges
        Iterable of edges. Each edge is presented as a tuple of two indices which
        tell the starting and ending note. The index correspond to location
        of point in points_list
    points
        Points location. Each point has x and y location.

    Returns
    -------
    The array with the hull/polygon points.
    """
    # sort based on first element of tuple to enable bisection search
    edges = sorted(edges, key=lambda x: x[0])
    # need to use all edges
    used = [False] * len(edges)

    # it is possible that we will find more separate hulls -
    # it happens in cases when a polygon has inner cycles
    polygons = []
    while not all(used):
        i = used.index(False)
        poly, new_used = _find_hull(edges, points, i)
        polygons.append(poly)
        used = [u1 or u2 for u1, u2 in zip(used, new_used)]

    # select polygon that is outside - the widest and the highest
    height_width = [np.sum(p.max(axis=0) - p.min(axis=0)) for p in polygons]
    i = height_width.index(max(height_width))
    return polygons[i]


def _get_shape_around_points(pts: np.ndarray, eps: Optional[float]) -> np.ndarray:
    """
    Compute the shape (concave hull) of a set of a cluster.
    """

    def add_edge(edges_list, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already. Remove the lines that are not outer
        edges - when (j, i) already in list means that two triangles
        has same edge what means it is not an outer edge
        """
        if (j, i) in edges_list:
            # both neighboring triangles are in shape - it's not a boundary edge
            edges_list.remove((j, i))
        else:
            edges_list.add((i, j))

    if len(pts) < 4:
        rng = list(range(len(pts)))
        return _edges_to_polygon(zip(rng, rng[1:] + rng[:1]), pts)

    eps = eps * 2 if eps else float("inf")
    tri = Delaunay(pts)
    edges = set()
    # loop over triangles: ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = pts[ia]
        pb = pts[ib]
        pc = pts[ic]

        # Lengths of sides of triangle
        a = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # filter - the longest edge of triangle must be smaller than epsilon
        if max(a, b, c) <= eps:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)

    return _edges_to_polygon(edges, pts)


def _smoothen_hull(
    hull: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float
) -> np.ndarray:
    """
    Expand hull a bit away from the cluster and smoothen it
    """
    # scale hull between 0 and 1 in both directions - help to produce the hull
    # with equal distance from the cluster when axes have different scaling
    x, y = hull[:, 0:1], hull[:, 1:2]
    x_diff, y_diff = (x_max - x_min), (y_max - y_min)
    hull = np.hstack(((x - x_min) / x_diff, (y - y_min) / y_diff))

    # buffer the line for 3 * dist_from_cluster and move it back for
    # 2 * dist_from_cluster. It will make a hull smother.

    # pyclipper work with integer so points need to be scaled first
    scaling_factor = 1000
    dist_from_cluster = 0.03
    scaled_hull = pyclipper.scale_to_clipper(hull, scaling_factor)

    # buffer the hull for dist_from_cluster * 3
    pco1 = pyclipper.PyclipperOffset()
    pco1.AddPath(scaled_hull, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    im_solution = pco1.Execute(dist_from_cluster * scaling_factor * 3)

    if len(im_solution) == 0:
        # when the solution is empty it means that polygon either consist of
        # one point, two points or all points are in line
        # ET_CLOSEDPOLYGON can't offset those polygons but ET_OPENROUND can
        pco1 = pyclipper.PyclipperOffset()
        pco1.AddPath(scaled_hull, pyclipper.JT_ROUND, pyclipper.ET_OPENROUND)
        im_solution = pco1.Execute(dist_from_cluster * scaling_factor * 3)

    # buffer the hull for dist_from_cluster * -2
    pco2 = pyclipper.PyclipperOffset()
    pco2.AddPath(im_solution[0], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution = pyclipper.scale_from_clipper(
        pco2.Execute(dist_from_cluster * scaling_factor * (-2)), scaling_factor
    )
    solution = np.array(solution).reshape(-1, 2)

    # scale the hue back for its original space
    return np.hstack(
        (solution[:, 0:1] * x_diff + x_min, solution[:, 1:2] * y_diff + y_min)
    )


def _global_range(points):
    """ Compute global min and max - for equal offsetting in each direction"""
    x, y = points[:, 0], points[:, 1]
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    # when x_max and x_min are equal set them 1 unit apart -
    # same behaviour as in scatter plot
    if x_max - x_min == 0:
        x_min -= 0.5
        x_max += 0.5
    if y_max - y_min == 0:
        y_min -= 0.5
        y_max += 0.5
    return x_min, x_max, y_min, y_max


def compute_concave_hulls(
    embedding: np.ndarray, clusters: np.ndarray, epsilon: Optional[float] = None
) -> Dict[int, np.ndarray]:
    """
    Function computes the points of the concave hull around points.

    Parameters
    ----------
    embedding
        Visualisation coordinates - embeddings
    clusters
       Cluster indices for each item.
    epsilon
        Points with distance > 2*epsilon are not connected resulting in concave
        hull. The parameter is optional, setting it to None result in convex hull.

    Returns
    -------
    The points of the concave hull. Dictionary with cluster index
    as a key and array of points as a value - [[x1, y1], [x2, y2], [x3, y3], ...]
    """
    assert embedding.shape[1] == 2, "Embedding must have two columns"
    embedding = embedding.astype(float)  # unique with axis doesn't work with object dtype
    x_min, x_max, y_min, y_max = _global_range(embedding)

    hulls = {}
    for cl in set(clusters) - {-1}:
        points = embedding[clusters == cl]

        # subsample when more than 1000 points - to speed up the algorithm
        if points.shape[0] > 1000:
            points = points[np.random.randint(points.shape[0], size=1000), :]

        # remove duplicates
        points = np.unique(points, axis=0)

        # compute hull around the cluster
        hull = _get_shape_around_points(points, epsilon)
        # smoothen and extend the hull
        hulls[cl] = _smoothen_hull(hull, x_min, x_max, y_min, y_max)
    return hulls


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    table = Table("iris")
    clusters = np.array([1] * 50 + [2] * 50 + [3] * 50)

    hulls = compute_concave_hulls(table.X[:1, :2], clusters[:1])
    plt.plot(table.X[:, 0], table.X[:, 1], "o")
    for k, h in hulls.items():
        plt.plot(h[:, 0], h[:, 1])
    plt.show()
