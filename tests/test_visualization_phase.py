import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from active_learning.phase_diagram import plot_phase_diagram
from active_learning.visualization import (
    magnetization_to_hsl_rgb,
    visualize_reconstruction_components,
)


def test_hsl_out_of_plane_field_is_not_false_rainbow():
    field = np.zeros((16, 16, 3), dtype=np.float32)
    field[:, :, 2] = 1.0

    rgb = magnetization_to_hsl_rgb(field)

    assert rgb.shape == field.shape
    assert np.allclose(rgb[:, :, 0], rgb[:, :, 1])
    assert np.allclose(rgb[:, :, 1], rgb[:, :, 2])


def test_phase_landscape_plot_smoke(tmp_path):
    df = pd.DataFrame({
        "Tx_val": [10e-9, 10e-9, 30e-9, 30e-9, 20e-9],
        "Tz_val": [10e-9, 30e-9, 10e-9, 30e-9, 20e-9],
        "MeanMz_abs": [0.95, 0.65, 0.45, 0.15, 0.5],
    })

    out_path = tmp_path / "phase.png"
    fig, _ = plot_phase_diagram(df, save_path=out_path)
    plt.close(fig)

    assert out_path.exists()


def test_component_reconstruction_plot_smoke(tmp_path):
    target = np.zeros((3, 16, 16), dtype=np.float32)
    prediction = target.copy()
    prediction[2] = 0.25

    out_path = tmp_path / "components.png"
    fig, _ = visualize_reconstruction_components(
        target,
        prediction,
        Tx=20e-9,
        Tz=30e-9,
        save_path=out_path,
    )
    plt.close(fig)

    assert out_path.exists()
