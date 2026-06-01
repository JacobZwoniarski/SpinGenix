import numpy as np

from .registry import PARAM_COLUMNS_V1, compute_param_hash


def _is_too_close(point, points, min_distance):
    if points is None:
        return False

    point_arr = np.array(point, dtype=float)
    for other in points:
        other_arr = np.array(other, dtype=float)
        if np.allclose(point_arr, other_arr, rtol=0.0, atol=1e-15):
            return True
        if min_distance > 0 and np.linalg.norm(point_arr - other_arr) < min_distance:
            return True
    return False


def select_top_k(
    U,
    Tx_grid,
    Tz_grid,
    K=10,
    min_distance=0.0,
    existing_points=None,
    excluded_hashes=None,
    param_columns=PARAM_COLUMNS_V1,
):
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
    excluded_hashes = excluded_hashes or set()

    for idx in flat_indices:
        if len(new_Tx) >= K:
            break

        i = idx // len(Tz_grid)
        j = idx % len(Tz_grid)

        Tx = Tx_grid[i]
        Tz = Tz_grid[j]

        point = [Tx, Tz]
        point_hash = compute_param_hash(
            {"Tx_val": Tx, "Tz_val": Tz},
            param_names=param_columns,
        )

        # optional: enforce distance from selected and already simulated points
        if point_hash in excluded_hashes:
            continue
        if _is_too_close(point, used_points, min_distance):
            continue
        if _is_too_close(point, existing_points, min_distance):
            continue

        new_Tx.append(float(Tx))
        new_Tz.append(float(Tz))
        used_points.append(point)

    return new_Tx, new_Tz
