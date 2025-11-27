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
    match = re.search(r'(\d+(?:\.\d+)?e-\d+)', s)
    return float(match.group(1)) if match else None


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

def resample_field_to_200(m, job):
    """
    m : raw magnetization array from zarr (t,x,y,z,3)
    returns field (200,200,3)
    """

    fo = m[0]   # (x,y,z,3)
    print("fo shape:", fo.shape)

    # dfield expects (nx, ny, nz, 3)
    field_arr = fo   # shape already (x,y,z,3)

    # build mesh
    nx, ny, nz = field_arr.shape[:3]
    cell = (job.Tx/nx, job.Ty/ny, job.Tz/nz)
    mesh = dfield.Mesh(p1=(0,0,0), p2=(job.Tx, job.Ty, job.Tz), cell=cell)

    m_field = dfield.Field(mesh, nvdim=3, value=field_arr, norm=1)

    # resample to 200×200×1 (z collapsed)
    resampled_field = m_field.resample((200,200,1)).array  # (200,200,1,3)

    # drop z dimension
    resampled_field = resampled_field[:,:,0,:]  # (200,200,3)

    return resampled_field


# ------------------------------------------------------------
# State classification logic
# ------------------------------------------------------------

def classify_state(mx, my, mz, b, Q):
    mean_mz = abs(np.mean(mz))

    if b > 0.2:
        return "in-plane"
    if b < 0.2 and mean_mz > 0.65:
        return "out-of-plane"
    if abs(Q) > 0.1:
        return "vortex"
    return "domain-wall"


# ------------------------------------------------------------
# Main preprocessing function
# ------------------------------------------------------------

def preprocess_simulation(zarr_path):
    """
    Reads one simulation's .zarr and returns:
        field_200x200x3,
        (Tx, Tz),
        metadata_dict
    """

    job = Pyzfn(zarr_path)
    store = zarr.open(zarr_path, mode="r")

    # magnetization field (t, nx, ny, nz, 3)
    m = store["m_relaxed"][:]

    # 1) Resample to target resolution
    field = resample_field_to_200(m, job)    # (200,200,3)

    # 2) Extract magnetization components
    mx = field[:,:,0]
    my = field[:,:,1]
    mz = np.abs(field[:,:,2])

    # 3) Compute physics features
    mean_mx = float(np.mean(mx))
    mean_my = float(np.mean(my))
    mean_mz = float(np.mean(mz))
    b = float(np.sqrt(mean_mx**2 + mean_my**2))

    Q = compute_topological_charge(mx, my, mz)

    # 4) Classify texture
    state = classify_state(mx, my, mz, b, Q)

    # 5) Extract parameters
    Tx = float(job.Tx)
    Tz = float(job.Tz)

    print("Raw field shape:", m.shape)
    print("Single t-step shape:", m[0].shape)

    metadata = {
        "Q": Q,
        "b": b,
        "MeanMx": mean_mx,
        "MeanMy": mean_my,
        "MeanMz": mean_mz,
        "State": state,
        "Aex": float(job.Aex),
        "Msat": float(job.Msat)
    }

    return field, (Tx, Tz), metadata