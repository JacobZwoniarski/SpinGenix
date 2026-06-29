import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from active_learning.phase_diagram import (
    plot_dataset_phase_comparison,
    plot_dataset_phase_diagram,
    plot_phase_diagram,
)
from active_learning.visualization import (
    _mask_prediction_to_reference,
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


def test_dataset_phase_plot_pads_full_parameter_ranges(tmp_path):
    df = pd.DataFrame({
        "Tx_val": [20e-9, 40e-9, 60e-9],
        "Tz_val": [20e-9, 50e-9, 90e-9],
        "MeanMz_abs": [0.85, 0.45, 0.12],
    })

    out_path = tmp_path / "phase_dataset.png"
    fig, ax = plot_dataset_phase_diagram(
        df,
        tx_range_nm=(10, 110),
        tz_range_nm=(10, 110),
        save_path=out_path,
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.close(fig)

    assert out_path.exists()
    assert xlim[0] < 10
    assert xlim[1] > 110
    assert ylim[0] < 10
    assert ylim[1] > 110


def test_dataset_phase_comparison_plot_smoke(tmp_path):
    df = pd.DataFrame({
        "Tx_val": [10e-9, 20e-9, 50e-9, 80e-9, 110e-9],
        "Tz_val": [10e-9, 80e-9, 50e-9, 20e-9, 110e-9],
        "MeanMz_abs": [0.9, 0.7, 0.45, 0.2, 0.1],
    })

    out_path = tmp_path / "phase_dataset_comparison.png"
    fig, axes = plot_dataset_phase_comparison(
        df,
        tx_range_nm=(10, 110),
        tz_range_nm=(10, 110),
        save_path=out_path,
    )
    plt.close(fig)

    assert out_path.exists()
    assert len(axes) == 2


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


def test_prediction_mask_zeros_background_padding():
    original = np.zeros((8, 8, 3), dtype=np.float32)
    original[2:6, 2:6, 2] = 1.0
    predicted = np.ones_like(original)

    masked = _mask_prediction_to_reference(original, predicted)

    assert np.all(masked[:2] == 0.0)
    assert np.all(masked[:, :2] == 0.0)
    assert np.all(masked[2:6, 2:6] == 1.0)
