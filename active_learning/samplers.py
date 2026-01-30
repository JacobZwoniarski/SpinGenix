import numpy as np

def _lhs_unit(n: int, d: int, seed: int = 0) -> np.ndarray:
    """
    Prosty Latin Hypercube Sampling w [0,1]^d bez zależności od SciPy.
    """
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, d))
    a = cut[:n]
    b = cut[1:n+1]
    rd = a[:, None] + u * (b - a)[:, None]  # (n, d) ale strata jest wspólna
    # permutacja w każdej kolumnie
    H = np.empty_like(rd)
    for j in range(d):
        perm = rng.permutation(n)
        H[:, j] = rd[perm, j]
    return H

def _try_sobol_unit(n: int, d: int, seed: int = 0, scramble: bool = True) -> np.ndarray | None:
    """
    Próbuje użyć SciPy Sobol. Jeśli SciPy nie ma – zwraca None.
    """
    try:
        from scipy.stats import qmc  # type: ignore
        # Sobol najlepiej działa dla n będącego potęgą 2, ale SciPy pozwala też na random_base2.
        m = int(np.ceil(np.log2(max(n, 2))))
        engine = qmc.Sobol(d=d, scramble=scramble, seed=seed)
        X = engine.random_base2(m=m)  # 2^m próbek
        return X[:n]
    except Exception:
        return None

def sample_params(
    bounds: dict[str, tuple[float, float]],
    n: int,
    method: str = "sobol",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Zwraca dict {param_name: np.ndarray shape (n,)} w zadanych bounds.
    method: "sobol" (preferowane) lub "lhs"
    """
    keys = list(bounds.keys())
    d = len(keys)

    if method.lower() == "sobol":
        X = _try_sobol_unit(n, d, seed=seed, scramble=True)
        if X is None:
            X = _lhs_unit(n, d, seed=seed)  # fallback
    elif method.lower() == "lhs":
        X = _lhs_unit(n, d, seed=seed)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    out: dict[str, np.ndarray] = {}
    for j, k in enumerate(keys):
        lo, hi = bounds[k]
        out[k] = lo + X[:, j] * (hi - lo)
    return out

def make_bounds_tx_tz(
    Tx_range: tuple[float, float],
    Tz_range: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    return {"Tx": Tx_range, "Tz": Tz_range}
