import numpy as np


def _is_too_close(point, points, min_distance):
    if points is None or min_distance <= 0:
        return False

    for other in points:
        if np.linalg.norm(np.array(point) - np.array(other)) < min_distance:
            return True
    return False


def select_top_k(U, Tx_grid, Tz_grid, K=10, min_distance=0.0, existing_points=None):
    """
    Selects top K parameter points with highest uncertainty.
    Returns:
        new_Tx, new_Tz  (lists of floats)
    """

    # Flatten the map
    flat_indices = np.argsort(U.flatten())[::-1]  # descending order

    new_Tx = []
    new_Tz = []

    used_points = []

    for idx in flat_indices:
        if len(new_Tx) >= K:
            break

        i = idx // len(Tz_grid)
        j = idx % len(Tz_grid)

        Tx = Tx_grid[i]
        Tz = Tz_grid[j]

        point = [Tx, Tz]

        # optional: enforce distance from selected and already simulated points
        if _is_too_close(point, used_points, min_distance):
            continue
        if _is_too_close(point, existing_points, min_distance):
            continue

        new_Tx.append(float(Tx))
        new_Tz.append(float(Tz))
        used_points.append(point)

    return new_Tx, new_Tz
