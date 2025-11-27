import numpy as np


def select_top_k(U, Tx_grid, Tz_grid, K=10, min_distance=0.0):
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

        # optional: enforce a minimum distance between selected points
        if min_distance > 0:
            too_close = False
            for p in used_points:
                if np.linalg.norm(np.array([Tx, Tz]) - np.array(p)) < min_distance:
                    too_close = True
                    break
            if too_close:
                continue

        new_Tx.append(float(Tx))
        new_Tz.append(float(Tz))
        used_points.append([Tx, Tz])

    return new_Tx, new_Tz