# VAE Phase Classification

Projekt do rekonstrukcji i klasyfikacji faz magnetycznych za pomocą warunkowego wariacyjnego autoenkodera (Conditional VAE).  
Wspiera analizę wyników symulacji mikromagnetycznych – predykcję i rekonstrukcję struktur spinowych na podstawie parametrów materiałowych oraz geometrycznych układu.

---

## Struktura projektu

- **src/** – kod źródłowy (architektura modelu, trening, ewaluacja, wizualizacja)
- **models/** – wytrenowane modele i checkpointy
- **losses/** – wykresy strat i logi z treningów
- **runs/** – logi do TensorBoard (do porównywania eksperymentów)
- **data/** – przykładowe/testowe pliki z danymi (nie commituj dużych plików!)
- **scripts/** – skrypty automatyzujące eksperymenty (np. `run_experiment.sh`)
- **requirements.txt** – wymagane biblioteki Python
- **LICENSE** – licencja projektu
- **README.md** – ten plik

---

## Szybki start

1. **Instalacja zależności**
    ```bash
    pip install -r requirements.txt
    pip install tensorboard
    ```

2. **Trenowanie modelu (lokalnie lub na serwerze, np. PCSS/H100):**
    ```bash
    python src/train.py \
      --meta /mnt/storage_2/scratch/pl0095-01/jakzwo/simulations/vx4/phase_classification_results_abs.h5 \
      --fields /mnt/storage_2/scratch/pl0095-01/jakzwo/simulations/vx4/phase_classification_fields_abs.npz \
      --batch_size 8 \
      --lr 0.0005 \
      --beta 0.01 \
      --num_epochs 100 \
      --latent_dim 128 \
      --save_path models/vae_h100_test.pth \
      --tensorboard \
      --logdir runs/h100_test
    ```
   *(Dostosuj ścieżki do własnych plików z danymi!)*

3. **Podgląd postępu w TensorBoard**

    - **Lokalnie:**
        ```bash
        tensorboard --logdir runs
        ```
        Otwórz [http://localhost:6006](http://localhost:6006) w przeglądarce.
    - **Na serwerze (PCSS, zdalnie):**
        1. Uruchom tensorboard na serwerze:  
           `tensorboard --logdir runs --port 6006`
        2. Przekieruj port:
            ```bash
            ssh -L 16006:localhost:6006 twoj_login@adres_serwera
            ```
        3. Wchodzisz lokalnie na: [http://localhost:16006](http://localhost:16006)

4. **Wizualizacja rekonstrukcji lub wykresów fazowych**
    ```bash
    python src/visualize.py --checkpoint models/vae_h100_test.pth --task phase
    python src/visualize.py --checkpoint models/vae_h100_test.pth --task sample --p 1.3 8.0 70 300
    ```

---

## Automatyzacja eksperymentów

- Skrypt `scripts/run_experiment.sh` umożliwia uruchamianie serii treningów z różnymi hiperparametrami.
- Każdy eksperyment zapisuje logi, modele i wykresy do osobnych katalogów (`runs/`, `models/`, `losses/`).

---

## Licencja

Projekt objęty licencją GNU GPL v3 (zobacz plik LICENSE).

---

**Autor:**  
Jakub Zwoniarski  
(kontakt: jakub.zwoniarski@gmail.com)

