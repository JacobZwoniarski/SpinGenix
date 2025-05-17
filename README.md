# VAE Phase Classification

Projekt do rekonstrukcji i klasyfikacji faz magnetycznych za pomocą warunkowego wariacyjnego autoenkodera (Conditional VAE).

## Struktura projektu

- **src/** – kod źródłowy (architektura modelu, trening, ewaluacja)
- **models/** – zapisane modele i checkpointy
- **data/** – sample/testowe pliki z danymi
- **experiments/** – wyniki eksperymentów, logi treningu, wykresy porównawcze
- **scripts/** – skrypty do uruchamiania eksperymentów (np. run_experiment.sh)
- **requirements.txt** – wymagane biblioteki
- **LICENSE** – licencja projektu

## Szybki start

1. **Instalacja zależności**
    ```bash
    pip install -r requirements.txt
    ```

2. **Uruchamianie treningu**
    - Z poziomu katalogu `src/`:
    ```bash
    python train.py --beta 0.01 --batch_size 8 --grid_size 200
    ```

3. **Automatyzacja eksperymentów**
    - Przykład uruchomienia kilku testów z różnymi parametrami:
    ```bash
    bash ../scripts/run_experiment.sh
    ```

## Automatyzacja eksperymentów

Skrypt `run_experiment.sh` w katalogu `scripts/` umożliwia uruchamianie serii eksperymentów z różnymi hiperparametrami. Wyniki każdego eksperymentu są zapisywane osobno.

## Zalecany workflow eksperymentów

- Każdy eksperyment na osobnym branchu GIT.
- Wyniki i logi zapisuj w `experiments/` – łatwo porównasz wersje.
- Po udanym eksperymencie scalaj do głównej gałęzi (`main` lub `master`).

## Licencja

Projekt objęty licencją GNU GPL v3 (zobacz plik LICENSE).

---

**Autor:**  
Jakub Zwoniarski  
(kontakt: jakub.zwoniarski@gmail.com)

