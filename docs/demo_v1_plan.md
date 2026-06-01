# SpinGenix Demo V1 Plan

Cel pierwszego demo: pokazać, że gotowy model 2-parametryczny przyjmuje
`Tx/Tz`, generuje kanoniczne pole magnetyzacji `200x200x3`, raportuje proste
metryki fizyczne i spina się z aktywnym uczeniem przez mapę fazową oraz CSV
akwizycji.

## Zakres Demo

1. Dataset readiness
   - liczba próbek i splity,
   - zakres `Tx/Tz`,
   - obecność `fields.npz` i normalizatora.

2. Model readiness
   - wybór checkpointu `param_surrogate.pt`,
   - zakres normalizatora checkpointu,
   - ostrzeżenie dla punktów poza zakresem treningowym modelu.

3. Field generation
   - presety: punkt wewnętrzny, cienka krawędź zakresu, wysoki `Tx`, wysoki
     `Tz`,
   - predykcja na CPU albo CUDA,
   - obraz HSL pola magnetyzacji,
   - metryki: `MeanMx`, `MeanMy`, `MeanMz_signed`, `MeanMz_abs`,
     `InPlaneOrder`, `Q`,
   - robocza klasyfikacja stanu.

4. Phase-map context
   - interpolowany diagram fazowy datasetu,
   - diagram modelu, jeśli checkpoint/run go wygenerował,
   - szybkie porównanie obszaru predykcji z obszarem bootstrapu.

5. Active-learning handoff
   - ostatni `acquisition_iterN.csv`,
   - liczba proponowanych punktów,
   - zakres nowych `Tx/Tz`,
   - ścieżki oczekiwanych `.zarr`.

## Kryteria Gotowości

- Platforma startuje bez dodatkowych zależności webowych:
  `.venv_sg/bin/python platform_app/server.py --host 127.0.0.1 --port 8765`.
- `/api/status` zwraca dataset, runy, checkpoint metadata i phase images.
- `/api/predict` zwraca PNG, metryki, zakres checkpointu i ostrzeżenia.
- UI ma czytelne przyciski na desktopie i mobile.
- Gotowy model po treningu można podmienić samym nowym runem pod `results/`,
  bez zmian w kodzie platformy.

## Po Demo V1

- Dodać nakładkę punktu `Tx/Tz` na diagramie fazowym.
- Dodać porównanie kilku checkpointów dla tego samego punktu.
- Dodać status Slurma dla aktywnych AL submitów.
- Dodać eksport pojedynczej predykcji do `.npz`/PNG.
