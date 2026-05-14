import numpy as np
from swapper import SimulationManager
from scipy.stats.qmc import LatinHypercube


def generate_params(N, Tx_range, Tz_range):
    """
    Generuje N par (Tx, Tz) za pomocą Latin Hypercube Sampling.
    
    Args:
        N: Liczba punktów do wygenerowania.
        Tx_range: Krotka (min, max) określająca przedział dla Tx.
        Tz_range: Krotka (min, max) określająca przedział dla Tz.
    
    Returns:
        Słownik z listami parametrów Tx i Tz.
    """
    lhs = LatinHypercube(d=2)
    samples = lhs.random(n=N)
    
    Tx = Tx_range[0] + samples[:, 0] * (Tx_range[1] - Tx_range[0])
    Tz = Tz_range[0] + samples[:, 1] * (Tz_range[1] - Tz_range[0])
    
    params = {
        "Tx": Tx.tolist(),
        "Tz": Tz.tolist()
    }
    return params

def add_points(existing_params, num_additional, Tx_range, Tz_range, oversample_factor=10):
    """
    Dodaje nowe punkty do istniejących, unikając pokrywania się z nimi.
    Używa LHS do wygenerowania większej liczby próbek i wybiera te, które są najbardziej oddalone od istniejących.
    
    Args:
        existing_params: Słownik z istniejącymi parametrami {"Tx": [...], "Tz": [...]}
        num_additional: Liczba nowych punktów do dodania (np. 100).
        Tx_range: Krotka (min, max) określająca przedział dla Tx.
        Tz_range: Krotka (min, max) określająca przedział dla Tz.
        oversample_factor: Mnożnik określający, ile więcej próbek wygenerować (domyślnie 10).
    
    Returns:
        Słownik z nowymi parametrami {"Tx": [...], "Tz": [...]}
    """
    # Skalowanie istniejących punktów do [0,1]
    existing_Tx = (np.array(existing_params["Tx"]) - Tx_range[0]) / (Tx_range[1] - Tx_range[0])
    existing_Tz = (np.array(existing_params["Tz"]) - Tz_range[0]) / (Tz_range[1] - Tz_range[0])
    existing_points = np.column_stack((existing_Tx, existing_Tz))
    
    # Generowanie nowych próbek LHS w przestrzeni [0,1]^2
    lhs = LatinHypercube(d=2)
    new_samples = lhs.random(n=oversample_factor * num_additional)
    
    # Obliczanie odległości między nowymi próbkami a istniejącymi punktami
    from scipy.spatial.distance import cdist
    distances = cdist(new_samples, existing_points)
    min_distances = np.min(distances, axis=1)
    
    # Wybór próbek z największymi minimalnymi odległościami
    sorted_indices = np.argsort(-min_distances)
    selected_indices = sorted_indices[:num_additional]
    selected_samples = new_samples[selected_indices]
    
    # Skalowanie wybranych próbek z powrotem do oryginalnych przedziałów
    Tx_selected = Tx_range[0] + selected_samples[:, 0] * (Tx_range[1] - Tx_range[0])
    Tz_selected = Tz_range[0] + selected_samples[:, 1] * (Tz_range[1] - Tz_range[0])
    
    additional_params = {
        "Tx": Tx_selected.tolist(),
        "Tz": Tz_selected.tolist()
    }
    return additional_params

# Przykład użycia
if __name__ == "__main__":
    # Definiujemy parametry
    N_initial = 400 # Początkowa liczba punktów
    #N_additional = 100  # Liczba dodatkowych punktów
    Tx_range = (10.0e-9, 100e-9)  # Przedział dla Tx
    Tz_range = (10.0e-9, 100e-9)  # Przedział dla Tz

    # Generujemy początkowe parametry
    params = generate_params(N_initial, Tx_range, Tz_range)
    print("Liczba początkowych punktów:", len(params["Tx"]))

    # # Inicjalizacja managera symulacji
    manager = SimulationManager(
        main_path="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/",
        destination_path="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/",
        prefix="vx5",
    )
    #print(params)
    # Uruchomienie początkowych symulacji
    manager.submit_all_simulations(
        params=params,
        last_param_name="Tz",
        minsim=0,
        maxsim=None,
        sbatch=1,
        pairs=True
    )

    # # # Generowanie 100 dodatkowych punktów
    # additional_params = add_points(params, N_additional, Tx_range, Tz_range)
    # print("Liczba dodatkowych punktów:", len(additional_params["Tx"]))

    # # Uruchomienie symulacji dla dodatkowych punktów
    # manager.submit_all_simulations(
    #     params=additional_params,
    #     last_param_name="Tz",
    #     minsim=0,
    #     maxsim=None,
    #     sbatch=False
    # )

    # # Opcjonalnie: połączenie wszystkich parametrów
    # for key in params:
    #     params[key].extend(additional_params[key])
    # print("Łączna liczba punktów:", len(params["Tx"]))
