# SpinGenix Project Handoff

Ten plik jest technicznym punktem odniesienia po uporządkowaniu V1. Ma pomóc
kontynuować pracę bez odtwarzania kontekstu z rozmowy.

## Aktualny Stan

- Repo jest na gałęzi `main`.
- `PLAN.md` jest ignorowany przez git i pozostaje lokalną notatką.
- Dane i wyniki (`data/`, `results/`, `*.h5`, `*.npz`, `*.csv`, `*.parquet`) są ignorowane przez git.
- Dotychczasowy dataset w `data/dataset` ma 53 próbki i jest zbyt wąski po osi `Tx`.
- Zakres obecnego datasetu:
  - `Tx = 10.193 .. 55.615 nm`
  - `Tz = 10.507 .. 98.537 nm`
- Katalog symulacji `simulations/vx5` został rozszerzony do planu 400 punktów:
  - ostatnio obserwowane: 400 punktów, 61 complete, 339 prepared `.mx3`
  - to jest za duże jak na docelowy bootstrap active learning
- Nie należy traktować modelu trenowanego na 53 punktach jako finalnego surrogate.

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
  --source initial_lhs_balanced
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
- najgorsze rekonstrukcje per split
- komponentowe PNG dla `Mx`, `My`, `Mz`, plus różnice

## Kryterium Przejścia Do Web App

Można przejść do lokalnej platformy webowej dopiero gdy:

- bootstrap obejmuje sensownie `Tx/Tz ~= 10..100/110 nm`
- holdouty są zamrożone i nie trafiają do treningu
- wykres fazowy datasetu ma sens fizyczny
- wykres fazowy modelu nie jest oczywistą ekstrapolacją/artefaktem
- rekonstrukcje komponentowe są akceptowalne przynajmniej na prostych fazach
- znamy ograniczenia modelu dla faz granicznych i symetrii `+Mz/-Mz`

Web app na tym etapie ma pokazywać:

- wejście `Tx/Tz`
- predykcję kanonicznego pola `200x200x3`
- fizyczny wymiar z metadata/parametrów, nie z rastra
- wykres fazowy datasetu
- wykres fazowy modelu w zakresie walidowanym
- ostrzeżenie, gdy użytkownik wychodzi poza zakres normalizatora

## Następne Implementacje

1. Dodać raport ewaluacyjny generujący automatycznie:
   - metryki per split
   - najgorsze przypadki
   - komponentowe rekonstrukcje
   - signed/absolute phase plots
2. Dodać jawny tryb budowania `vx5_bootstrap`, który wybiera zbalansowany mały
   subset i zapisuje manifest wybranych `param_hash`.
3. Dopiero potem poprawiać architekturę modelu.
4. Dopiero po stabilnym V1 bootstrapie zaczynać lokalną web app.
