# SpinGenix Project Handoff

Ten plik jest technicznym punktem odniesienia po uporządkowaniu V1. Ma pomóc
kontynuować pracę bez odtwarzania kontekstu z rozmowy.

## Aktualny Stan

- Repo jest na gałęzi `main`.
- `PLAN.md` jest ignorowany przez git i pozostaje lokalną notatką.
- Dane i wyniki (`data/`, `results/`, `*.h5`, `*.npz`, `*.csv`, `*.parquet`) są ignorowane przez git.
- Aktualny dataset w `data/dataset` ma 60 próbek i obejmuje bootstrap
  `vx5_bootstrap` po top-upie.
- Zakres aktualnego datasetu:
  - `Tx ~= 10.17 .. 101.89 nm`
  - `Tz ~= 10.45 .. 109.77 nm`
- Live katalog symulacji `simulations/vx5` jest oczyszczony do 60 kompletnych
  punktów bootstrapu. Przypadkowy plan 400 punktów jest zarchiwizowany, nie
  skasowany.
- Nie należy traktować checkpointów sprzed poprawki maskowanej norm penalty
  jako finalnego surrogate.

## Update 2026-05-24

- `data/dataset` został odbudowany po top-upie i zawiera 60 próbek.
- Live prefix `/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/vx5` jest
  oczyszczony do aktualnego bootstrapu: `points=60`, `complete=60`.
- Finalny `vx5_bootstrap` dataset:
  - split counts: `train=39`, `val=9`, `test_holdout=9`,
    `boundary_holdout=3`
  - registry strict holdout: OK
  - dataset split leakage: 0
  - `Tx (0,20] nm`: 14
  - `Tx (20,40] nm`: 10
  - `Tx (40,60] nm`: 6
  - `Tx (60,80] nm`: 12
  - `Tx (80,100] nm`: 10
  - `Tx (100,120] nm`: 8
  - `Tz (0,20] nm`: 8
  - `Tz (20,40] nm`: 10
  - `Tz (40,60] nm`: 9
  - `Tz (60,80] nm`: 10
  - `Tz (80,100] nm`: 16
  - `Tz (100,120] nm`: 7
- `scripts/build_dataset_from_zarr.py` respektuje teraz istniejące wpisy
  registry: wcześniejsze holdouty pozostają holdoutami, a wcześniejsze
  `train/val` nie mogą zostać nowym holdoutem. To usuwa konflikt
  `Strict holdout violation` i pilnuje leakage przed preprocessingiem.
- Kod diagnostyczny został poprawiony:
  - diagramy fazowe renderują domyślnie interpolowany `landscape` z punktami
    źródłowymi na wierzchu,
  - zapisywane są osobne diagramy `MeanMz_abs` i `MeanMz_signed`,
  - HSL nie używa już pełnej normy magnetyzacji jako saturacji, więc prawie
    jednorodne pola `+/-Mz` nie robią fałszywej tęczy,
  - rekonstrukcje zapisują komponentowe panele `Mx/My/Mz` z różnicą
    `prediction - target` i stabilną skalą kolorów,
  - `sample_param_surrogate_grid.py` domyślnie sampluje zakres normalizatora,
    a nie sztywny `Tx/Tz=10..100 nm`.
- Po pierwszym teście H100 naprawiono ważny błąd treningowy: kara normy
  magnetyzacji jest liczona tylko wewnątrz maski materiału z targetu, a nie
  w pustym tle rastra. Checkpoint sprzed tej poprawki należy traktować jako
  diagnostyczny/nieaktualny.
- `train_param_surrogate.py` zapisuje teraz audyt datasetu, modelowy phase
  diagram w zakresie normalizatora oraz zbalansowane rekonstrukcje z top-up
  `Tx>=50 nm` i starego low-`Tx`, zamiast pokazywać wyłącznie globalnie
  najgorsze przypadki.
- Główna pętla active learning nadal działa przez `run_active_learning.py`.
  Runner wymaga teraz jawnego `--dry-run` albo `--submit`, trenuje CVAE tylko
  na `split=train`, wyklucza `val/test_holdout/boundary_holdout` z treningu,
  używa registry jako exclusion setu w acquisition i zapisuje
  `results/.../acquisition/acquisition_iterN.csv` przed ewentualnym Slurmem.
- `wait_for_simulations` czeka teraz na kompletne `.zarr`, nie samo
  pojawienie się katalogu.
- Smoke-checki wykonane lokalnie:
  - `scripts/smoke_v1_pipeline.py --train-epochs 0`
  - `scripts/train_param_surrogate.py --epochs 0 --max-samples 4`
  - `scripts/sample_param_surrogate_grid.py --grid-points 4 --no-fields`
  - `.venv_sg/bin/python -m pytest tests` -> 3 passed

## Update 2026-06-01

- Pętla active learning została ponownie sprawdzona po stronie kodu:
  - `run_active_learning.py` ma jawny tryb `--dry-run` / `--submit`,
  - `training_subset()` trenuje tylko na `split=train`,
  - acquisition wyklucza punkty z datasetu i registry, w tym holdouty,
  - `wait_for_simulations()` sprawdza kompletność `.zarr`,
  - `SimulationManager` normalizuje `simulations_dir`, więc działa z końcowym
    slashem i bez niego,
  - acquisition zapisuje CSV także wtedy, gdy selekcja jest pusta.
- Dodane testy rdzenia AL w `tests/test_active_learning_core.py`.
- Testy lokalne:
  - `.venv_sg/bin/python -m py_compile ...` -> OK
  - `.venv_sg/bin/python -m pytest -p no:cacheprovider tests -q` -> 6 passed
  - CPU smoke `run_active_learning.py --dry-run --epochs 0 --grid-points 1`
    -> zakończony sukcesem, bez submitowania Slurma.
- Pierwsza wersja platformy webowej istnieje w `platform_app/`:
  - backend: `platform_app/server.py`, bez dodatkowych zależności webowych,
  - frontend: `platform_app/static/`,
  - widoki: dataset overview, phase plots, checkpoint prediction,
    active-learning acquisitions, runs,
  - endpointy: `/api/status`, `/api/file`, `/api/predict`.
- Platforma została sprawdzona lokalnie:
  - `/` -> 200
  - `/api/status` -> 200
  - `/api/predict` z checkpointu CPU -> 200, zwraca metryki i PNG base64.
- Platforma została dopracowana pod demo V1:
  - spokojniejszy, ciemny sidebar i neutralny workspace,
  - pionowa nawigacja bez ściskania tekstu przycisków,
  - widok `Demo` jako pierwszy ekran,
  - suwaki i pola liczbowe `Tx/Tz`,
  - presety punktów w zakresie checkpointu,
  - metadata checkpointu: zakres normalizatora, model config, rozmiar,
  - `/api/predict` zwraca teraz także `normalizer_range_nm`, `warnings`,
    `normalized_params`, `state_guess` oraz `InPlaneOrder`,
  - ostrzeżenia pojawiają się dla punktów poza zakresem treningowym
    checkpointu.
- Dokładny plan pierwszego demo jest w `docs/demo_v1_plan.md`.

Dataset jest gotowy do krótkiego/pełnego treningu 2-parametrycznego na H100.

### Cleanup Przed Treningiem

- `report_simulation_status.py --prefix vx5` pokazuje teraz tylko aktualny
  bootstrap:
  - `points=60`
  - `complete=60`
  - brak przygotowanych `.mx3` i `.mx3_status.*`
- Źródła obecnego datasetu zostały na miejscu; walidacja `source_path`
  dla 60 próbek przechodzi.
- Materiały z przypadkowego pełnego planu 400 punktów nie zostały skasowane,
  tylko przeniesione do:
  `/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/_archive/vx5_accidental_400_cleanup_20260524_184126`
  - `complete_unused_zarr`: 30
  - `prepared_mx3`: 310
  - `mx3_status`: 90
  - `sbatch_dir`: 1
  - `empty_dir`: 685
  - manifest: `cleanup_manifest.csv`
- Repozytoryjne logi `slurm-*.out` zostały przeniesione do:
  `/mnt/storage_5/scratch/pl0095-01/jakzwo/SpinGenix_remote/_archive/slurm_out_20260524_184126`

## Co Poszło Nie Tak

Pierwotny `vx5` bootstrap miał być mały, ale powinien obejmować zakres mniej
więcej do `Tx/Tz = 100..110 nm`. W praktyce początkowe 53 punkty kończyły się
przy `Tx ~= 55.6 nm`, więc późniejszy diagram modelu dla `Tx=100 nm` był
ekstrapolacją.

Następnie użyty został batchowy skrypt `submit_vx5_initial.py` z targetem 400.
To przygotowało/uruchomiło pełny plan LHS zamiast małego top-upu brakującego
zakresu `Tx=50..100 nm`. To nie niszczy danych, ale miesza koncepcyjnie etap
bootstrapu z gęstą mapą referencyjną.

## Zasada Od Teraz

- Bootstrap ma być mały i zbalansowany.
- Full 400 punktów jest opcjonalną mapą referencyjną, nie domyślnym trybem.
- Active learning ma dodawać małe paczki, typowo 20-50 punktów.
- Nie budować datasetu przez ślepe wzięcie wszystkich kompletnych `vx5`, jeśli
  w katalogu leżą symulacje z przypadkowo przygotowanego planu 400.
- Przed każdym treningiem sprawdzić rozkład `Tx/Tz` i splity.

## Bezpieczniki W Kodzie

`scripts/submit_vx5_initial.py` ma teraz bezpiecznik:

- jeśli `--submit` miałby wysłać więcej niż 50 jobów,
- i nie podano `--max-submit`,
- i nie podano `--allow-large-submit`,
- skrypt przerywa działanie.

Do naprawy bootstrapu dodany jest osobny planner:

```bash
.venv_sg/bin/python scripts/plan_vx5_topup.py --max-submit 30
```

Domyślnie wybiera małą partię z `Tx=50..100 nm`, rozłożoną po binach `Tx` i po
zakresie `Tz`. Nie wysyła jobów bez `--submit`.

## Protokół Naprawy Startowego Datasetu

1. Sprawdź, czy w Slurmie wiszą przypadkowe pending joby z pełnego planu:

```bash
squeue -u "$USER" -h -o "%i %j %t %R" | head -40
squeue -u "$USER" -h | wc -l
```

2. Jeśli jest bardzo dużo pending jobów `Tz_...` i chcesz wrócić do małego
   bootstrapu, anuluj tylko pendingi, nie runningi:

```bash
squeue -u "$USER" -h -t PD -o "%i %j" \
  | awk '$2 ~ /^Tz_/ {print $1}' \
  | xargs -r scancel
```

3. Nie kasuj kompletnych `.zarr`. To są użyteczne dane.

4. Zaplanuj mały top-up brakującego zakresu:

```bash
.venv_sg/bin/python scripts/plan_vx5_topup.py \
  --topup-tx-min-nm 50 \
  --topup-tx-max-nm 100 \
  --tx-bin-width-nm 10 \
  --target-per-tx-bin 6 \
  --max-submit 30
```

5. Jeśli lista wygląda sensownie i `squeue` nie pokazuje duplikatów, wyślij:

```bash
module load cuda/12.6.0_560.28.03

.venv_sg/bin/python scripts/plan_vx5_topup.py \
  --topup-tx-min-nm 50 \
  --topup-tx-max-nm 100 \
  --tx-bin-width-nm 10 \
  --target-per-tx-bin 6 \
  --max-submit 30 \
  --submit \
  --amumax-bin /mnt/storage_5/scratch/pl0095-01/bin/amumax \
  --cuda-module cuda/12.6.0_560.28.03
```

6. Monitoruj:

```bash
.venv_sg/bin/python scripts/report_simulation_status.py --prefix vx5 --limit 20
```

7. Gdy top-up się skończy, zbuduj mały zbalansowany dataset, nie wszystkie
   dostępne punkty:

```bash
.venv_sg/bin/python scripts/build_dataset_from_zarr.py \
  --raw-root /mnt/storage_5/scratch/pl0095-01/jakzwo/simulations \
  --dataset-mode vx5_bootstrap \
  --prefix vx5 \
  --out-dir data/dataset \
  --registry-dir data/registry \
  --bins 5 \
  --per-bin 4 \
  --auto-split \
  --val-fraction 0.15 \
  --test-holdout-fraction 0.15 \
  --boundary-holdout-fraction 0.05 \
  --split train \
  --source vx5_bootstrap \
  --manifest-path data/dataset/vx5_bootstrap_manifest.csv
```

8. Sprawdź rozkład datasetu:

```bash
.venv_sg/bin/python - <<'PY'
import pandas as pd
df = pd.read_hdf("data/dataset/meta.h5", "data")
print(len(df))
print((df[["Tx_val", "Tz_val"]] * 1e9).agg(["min", "max", "mean"]).round(3))
print(df["split"].value_counts())
print(pd.cut(df["Tx_val"] * 1e9, bins=[0,20,40,60,80,100,120]).value_counts().sort_index())
PY
```

## Trening I Ewaluacja Po Naprawie

Smoke:

```bash
.venv_sg/bin/python scripts/smoke_v1_pipeline.py \
  --raw-root /mnt/storage_5/scratch/pl0095-01/jakzwo/simulations \
  --prefix vx5 \
  --meta-path data/dataset/meta.h5 \
  --fields-path data/dataset/fields.npz \
  --normalizer-path data/dataset/param_normalizer.json \
  --out-dir results/v1_pipeline/smoke \
  --device auto \
  --train-epochs 0
```

Krótki baseline:

```bash
.venv_sg/bin/python scripts/train_param_surrogate.py \
  --meta-path data/dataset/meta.h5 \
  --fields-path data/dataset/fields.npz \
  --normalizer-path data/dataset/param_normalizer.json \
  --out-dir results/v1_pipeline/param_surrogate \
  --device auto \
  --epochs 50 \
  --batch-size 8
```

Jeśli `device: cpu`, to wyniki są smoke-testem, nie finalnym treningiem. Pełne
benchmarki uruchamiać na code serverze lub jobie z widocznym CUDA/H100.

## Active Learning Po Naprawie

Najpierw krótki dry-run na przydzielonym GPU, bez submitowania Slurma:

```bash
PYTHONUNBUFFERED=1 .venv_sg/bin/python -u run_active_learning.py \
  --dry-run \
  --iterations 1 \
  --epochs 1 \
  --grid-points 12 \
  --k-new 5 \
  --mc-samples 4 \
  --device cuda \
  --results-dir results/active_learning/dryrun_$(date +%Y%m%d_%H%M%S) \
  --acquisition-min-distance-nm 0
```

Jeśli dry-run pokaże sensowne punkty w `acquisition_iter1.csv`, puścić realną
iterację:

```bash
PYTHONUNBUFFERED=1 .venv_sg/bin/python -u run_active_learning.py \
  --submit \
  --iterations 1 \
  --epochs 20 \
  --grid-points 40 \
  --k-new 20 \
  --mc-samples 10 \
  --device cuda \
  --poll-interval 120 \
  --max-wait-hours 24 \
  --results-dir results/active_learning/run_$(date +%Y%m%d_%H%M%S) \
  --simulations-dir /mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/ \
  --simulation-prefix vxAL \
  --amumax-bin /mnt/storage_5/scratch/pl0095-01/bin/amumax \
  --cuda-module cuda/12.6.0_560.28.03
```

Nie odpalać już gołego `python run_active_learning.py`: tryb jest jawny, żeby
nie powtórzyć przypadkowego dużego submitu.

## Wizualizacje

Aktualne wnioski:

- `MeanMz_abs` jest dobry do historycznego wykresu fazowego, ale ukrywa znak
  `Mz`.
- `MeanMz_signed` trzeba raportować osobno, bo model może mylić symetryczne
  stany `+z` i `-z`.
- HSL jest przydatne jakościowo, ale może być mylące dla prawie jednorodnych
  pól out-of-plane.
- Do diagnostyki rekonstrukcji preferować komponentowe panele `Mx/My/Mz` oraz
  mapy różnic, nie tylko HSL.
- Diagram modelu wolno interpretować tylko w zakresie normalizatora/datasetu.
  Jeśli dataset kończy się przy `Tx=55 nm`, diagram do `Tx=100 nm` jest
  ekstrapolacją.

Minimalny zestaw po każdym treningu:

- phase dataset, kolor `MeanMz_abs`
- phase dataset, kolor `MeanMz_signed`
- phase model w zakresie datasetu, nie szerszym
- metryki per split
- `evaluation/dataset_audit.json` i biny `Tx/Tz` per split
- najgorsze rekonstrukcje per split
- komponentowe PNG dla `Mx`, `My`, `Mz`, plus różnice

## Platforma V1

Pierwsza lokalna platforma webowa jest dostępna. Uruchomienie:

```bash
cd /mnt/storage_5/scratch/pl0095-01/jakzwo/SpinGenix_remote
.venv_sg/bin/python platform_app/server.py --host 127.0.0.1 --port 8765
```

Jeśli odpalasz ją na węźle i chcesz wejść z laptopa, zrobić tunnel SSH do
portu `8765` albo uruchomić na `--host 0.0.0.0` tylko w środowisku, w którym
masz pewność co do dostępu sieciowego.

Platforma V1 pokazuje:

- status datasetu i splitów,
- dostępne checkpointy pod `results/`,
- interpolowane diagramy fazowe,
- podgląd rekonstrukcji/predykcji z checkpointu dla `Tx/Tz`,
- CSV z akwizycji active learning,
- listę runów i podstawowe metryki.

## Kryterium Pełniejszej Platformy

Przed traktowaniem platformy jako narzędzia publikacyjnego nadal sprawdzić:

- bootstrap obejmuje sensownie `Tx/Tz ~= 10..100/110 nm`
- holdouty są zamrożone i nie trafiają do treningu
- wykres fazowy datasetu ma sens fizyczny
- wykres fazowy modelu nie jest oczywistą ekstrapolacją/artefaktem
- rekonstrukcje komponentowe są akceptowalne przynajmniej na prostych fazach
- znamy ograniczenia modelu dla faz granicznych i symetrii `+Mz/-Mz`

## Następne Implementacje

1. Dodać do platformy ostrzeżenie, gdy `Tx/Tz` wychodzi poza zakres
   normalizatora checkpointu.
2. Dodać widok statusu Slurma/AL: submitted, running, done, failed,
   incomplete `.zarr`.
3. Dodać porównanie kilku checkpointów na tym samym `Tx/Tz`.
4. Po pierwszym pełnym AL runie ocenić, czy architektura CVAE wystarcza, czy
   trzeba przejść na silniejszy conditional decoder/diffusion-style surrogate.
