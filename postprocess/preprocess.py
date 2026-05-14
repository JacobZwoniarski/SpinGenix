import os
import re
import numpy as np
import zarr
from pyzfn import Pyzfn
import discretisedfield as dfield

# ------------------------------------------------------------
# Parse scientific notation from folder/file names
# ------------------------------------------------------------

def parse_scientific_notation(s: str) -> float:
    match = re.search(r'(\d+(?:\.\d+)?e[+-]?\d+)', s, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _as_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _job_attr(job, name, default=None):
    if hasattr(job, name):
        return getattr(job, name)

    attrs = getattr(job, "attrs", None)
    if attrs is not None and name in attrs:
        return attrs[name]

    return default


def _store_attr(store, name, default=None):
    try:
        return store.attrs.get(name, default)
    except Exception:
        return default


def _param_from_path(zarr_path, name):
    pattern = re.compile(rf"{name}_?(\d+(?:\.\d+)?e[+-]?\d+)", re.IGNORECASE)
    for part in reversed(os.path.normpath(zarr_path).split(os.sep)):
        match = pattern.search(part)
        if match:
            return float(match.group(1))
    return None


def _get_attr(job, store, name, default=None):
    value = _job_attr(job, name, None)
    if value is not None:
        return value
    return _store_attr(store, name, default)


def _orient_field_array(field_arr, job, store):
    """
    Return magnetization in discretisedfield's expected (nx, ny, nz, 3) order.

    Amumax/MuMax zarr files found in this project are commonly stored as
    (nz, ny, nx, 3), while older code sometimes treated them as (nx, ny, nz, 3).
    We use Nx/Ny/Nz metadata when available and fall back to the common
    (z, y, x, 3) layout for thin-film arrays.
    """
    if field_arr.ndim != 4 or field_arr.shape[-1] != 3:
        raise ValueError(
            f"Expected a single time-step with shape (?,?,?,3), got {field_arr.shape}"
        )

    nx = _as_int(_get_attr(job, store, "Nx"))
    ny = _as_int(_get_attr(job, store, "Ny"))
    nz = _as_int(_get_attr(job, store, "Nz"))

    if nx and ny and nz:
        if field_arr.shape[:3] == (nx, ny, nz):
            return field_arr
        if field_arr.shape[:3] == (nz, ny, nx):
            return np.transpose(field_arr, (2, 1, 0, 3))
        if field_arr.shape[:3] == (ny, nx, nz):
            return np.transpose(field_arr, (1, 0, 2, 3))

    if field_arr.shape[0] <= 4 and field_arr.shape[1] == field_arr.shape[2]:
        return np.transpose(field_arr, (2, 1, 0, 3))

    return field_arr


# ------------------------------------------------------------
# Skyrmion number (topological charge)
# ------------------------------------------------------------

def compute_topological_charge(mx, my, mz, dx=1.0, dy=1.0):
    dmx_dx = (np.roll(mx, -1, axis=1) - np.roll(mx, 1, axis=1)) / (2 * dx)
    dmy_dx = (np.roll(my, -1, axis=1) - np.roll(my, 1, axis=1)) / (2 * dx)
    dmz_dx = (np.roll(mz, -1, axis=1) - np.roll(mz, 1, axis=1)) / (2 * dx)

    dmx_dy = (np.roll(mx, -1, axis=0) - np.roll(mx, 1, axis=0)) / (2 * dy)
    dmy_dy = (np.roll(my, -1, axis=0) - np.roll(my, 1, axis=0)) / (2 * dy)
    dmz_dy = (np.roll(mz, -1, axis=0) - np.roll(mz, 1, axis=0)) / (2 * dy)

    cross_x = dmy_dx * dmz_dy - dmz_dx * dmy_dy
    cross_y = dmz_dx * dmx_dy - dmx_dx * dmz_dy
    cross_z = dmx_dx * dmy_dy - dmy_dx * dmx_dy

    q = mx * cross_x + my * cross_y + mz * cross_z
    Q = np.sum(q) * dx * dy / (4 * np.pi)
    return float(Q)


# ------------------------------------------------------------
# Resampling arbitrary mesh → fixed (200×200×3)
# ------------------------------------------------------------

def resample_field_to_200(m, job, store=None, target_size=(200, 200)):
    """
    m : raw magnetization array from zarr (usually t,z,y,x,3)
    returns field (200,200,3)
    """

    if store is None:
        store = {}

    fo = m[0]
    field_arr = _orient_field_array(fo, job, store)
    nx, ny, nz = field_arr.shape[:3]

    Tx = _as_float(_get_attr(job, store, "Tx"))
    Ty = _as_float(_get_attr(job, store, "Ty"), Tx)
    Tz = _as_float(_get_attr(job, store, "Tz"))

    if Tx is None or Ty is None or Tz is None:
        raise ValueError("Missing Tx/Ty/Tz metadata; cannot build discretisedfield mesh")

    cell = (Tx / nx, Ty / ny, Tz / nz)
    mesh = dfield.Mesh(p1=(0, 0, 0), p2=(Tx, Ty, Tz), cell=cell)

    m_field = dfield.Field(mesh, nvdim=3, value=field_arr, norm=1)

    # Resample to a fixed 2D representation; collapse z to one layer.
    resampled_field = m_field.resample((target_size[0], target_size[1], 1)).array
    resampled_field = resampled_field[:, :, 0, :]

    return resampled_field.astype(np.float32)


# ------------------------------------------------------------
# State classification logic
# ------------------------------------------------------------

def classify_state(mx, my, mz, b, Q):
    mean_mz_abs = float(np.mean(np.abs(mz)))

    if b > 0.2:
        return "in-plane"
    if b < 0.2 and mean_mz_abs > 0.65:
        return "out-of-plane"
    if abs(Q) > 0.1:
        return "vortex"
    return "domain-wall"


# ------------------------------------------------------------
# Main preprocessing function
# ------------------------------------------------------------

def preprocess_simulation(zarr_path, target_size=(200, 200), verbose=False):
    """
    Reads one simulation's .zarr and returns:
        field_200x200x3,
        (Tx, Tz),
        metadata_dict
    """

    job = Pyzfn(zarr_path)
    store = zarr.open(zarr_path, mode="r")

    if "m_relaxed" not in store:
        raise KeyError(f"Missing 'm_relaxed' in {zarr_path}")

    # magnetization field (t, nx, ny, nz, 3)
    m = store["m_relaxed"][:]

    # 1) Resample to target resolution
    field = resample_field_to_200(m, job, store=store, target_size=target_size)

    # 2) Extract magnetization components. Keep signed mz for topology and
    # store both signed and absolute means for downstream physics/plots.
    mx = field[:, :, 0]
    my = field[:, :, 1]
    mz = field[:, :, 2]
    mz_abs = np.abs(mz)

    # 3) Compute physics features
    mean_mx = float(np.mean(mx))
    mean_my = float(np.mean(my))
    mean_mz_signed = float(np.mean(mz))
    mean_mz_abs = float(np.mean(mz_abs))
    b = float(np.sqrt(mean_mx**2 + mean_my**2))

    Q = compute_topological_charge(mx, my, mz)

    # 4) Classify texture
    state = classify_state(mx, my, mz, b, Q)

    # 5) Extract parameters
    Tx = _as_float(_get_attr(job, store, "Tx"), _param_from_path(zarr_path, "Tx"))
    Ty = _as_float(_get_attr(job, store, "Ty"), Tx)
    Tz = _as_float(_get_attr(job, store, "Tz"), _param_from_path(zarr_path, "Tz"))

    if Tx is None or Tz is None:
        raise ValueError(f"Could not extract Tx/Tz from metadata or path: {zarr_path}")

    if verbose:
        print("Raw field shape:", m.shape)
        print("Single t-step shape:", m[0].shape)
        print("Resampled field shape:", field.shape)

    metadata = {
        "Q": Q,
        "b": b,
        "MeanMx": mean_mx,
        "MeanMy": mean_my,
        "MeanMz_signed": mean_mz_signed,
        "MeanMz_abs": mean_mz_abs,
        "MeanMz": mean_mz_abs,
        "State": state,
        "Aex": _as_float(_get_attr(job, store, "Aex")),
        "Msat": _as_float(_get_attr(job, store, "Msat")),
        "Ty_val": Ty,
        "Nx": _as_int(_get_attr(job, store, "Nx")),
        "Ny": _as_int(_get_attr(job, store, "Ny")),
        "Nz": _as_int(_get_attr(job, store, "Nz")),
        "dx": _as_float(_get_attr(job, store, "dx")),
        "dy": _as_float(_get_attr(job, store, "dy")),
        "dz": _as_float(_get_attr(job, store, "dz")),
        "target_Nx": int(target_size[0]),
        "target_Ny": int(target_size[1]),
        "target_Nz": 1,
        "source_path": os.path.abspath(zarr_path),
    }

    return field, (Tx, Tz), metadata
