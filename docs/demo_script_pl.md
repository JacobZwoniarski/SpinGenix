# Scenariusz Demo SpinGenix V1

## Cel narracji

Pokazujemy pierwszą wersję konsoli SpinGenix dla modelu dwuparametrycznego `Tx/Tz`. To jest demonstracja przepływu: dataset, checkpoint, predykcja pola, kontekst fazowy i handoff do active learning. Nie sprzedajemy tego jako finalnego modelu po wielu iteracjach AL.

## Start

1. Otwórz konsolę demo.
2. Zostań na widoku `Demo`.
3. Powiedz:
   „To jest pierwsza wersja konsoli SpinGenix dla modelu dwuparametrycznego `Tx/Tz`. Na tym etapie pokazujemy pełny przepływ interakcji z modelem i datasetem, a nie jeszcze finalną jakość modelu po wielu iteracjach active learning.”

## Dataset i splity

1. Pokaż `System Snapshot` oraz badge `60 samples` w panelu bocznym.
2. Przejdź na `Dataset`, jeśli trzeba pokazać szczegóły splitów.
3. Powiedz:
   „Dataset startowy ma 60 próbek. Splity są jawne: 39 train, 9 val, 9 test holdout i 3 boundary holdout. Holdouty są zamrożone, więc active learning nie może ich użyć jako nowych punktów treningowych. To zabezpiecza przed leakage.”

## Zakres checkpointu

1. W panelu bocznym pokaż `Tx/Tz envelope`.
2. Ustaw preset `Interior`.
3. Powiedz:
   „Platforma pilnuje zakresu normalizatora checkpointu. Jeśli wybierzemy punkt poza obszarem uczenia, backend zwróci ostrzeżenie zamiast udawać, że predykcja ma taki sam poziom zaufania.”

## Live prediction

1. Kliknij `Interior`, potem `Generate Field`.
2. Pokaż komponenty `Mx/My/Mz` i metryki fizyczne.
3. Przetestuj jeszcze `High Tx` albo `High Tz`.
4. Powiedz:
   „Dla wybranego punktu model generuje kanoniczne pole `200x200x3`, a aplikacja od razu liczy proste deskryptory: średnie komponenty magnetyzacji, uporządkowanie w płaszczyźnie i ładunek topologiczny `Q`.”

## Phase context

1. Wróć do `Demo` albo przejdź na `Phase Map`.
2. Pokaż mapę `Dataset |MeanMz|` jako główny kontekst.
3. Powiedz:
   „Ten widok pokazuje, gdzie leży wybrany punkt w przestrzeni parametrów. Interpolacja jest pomocnicza; punkty źródłowe są ważniejsze, bo pokazują, gdzie mamy realne symulacje.”
4. Jeśli ktoś pyta o kolory:
   „Dla `MeanMz_abs` używamy spokojnej palety sekwencyjnej, a dla `MeanMz_signed` symetrycznej palety z centrum w zerze.”

## Rekonstrukcje

1. Przejdź na `Prediction Detail`.
2. Pokaż `Reconstruction Checks`.
3. Powiedz:
   „Do demo preferujemy komponenty `Mx/My/Mz`, bo HSL bywa mylący przy prawie jednorodnych stanach. Tu ważniejsza jest konsekwencja prezentacji niż efektowna kolorystyka.”

## Active learning

1. Przejdź na `Active Learning`.
2. Pokaż najnowszy `acquisition_iter1.csv`.
3. Powiedz:
   „Dry-run AL na H100 działa: trenuje model, liczy uncertainty i wybiera kolejne punkty. Wybrane punkty są poza datasetem i registry, a holdout nie jest używany do acquisition.”

## Blocker HPC

Powiedz krótko:

„Sama pętla AL dochodzi do wyboru nowych punktów. Obecny blocker jest integracyjny: na tym HPC dopinamy właściwy sposób uruchamiania Amumax przez kontener, wrapper albo moduł. Walidacja zatrzymuje submit przed `sbatch`, więc to nie jest problem modelu ani selekcji punktów.”

Nie rozwijaj:

- ścieżek do brakującej binarki,
- pełnych tracebacków,
- szczegółów Slurma, jeśli nikt nie pyta.

## Zakończenie

Powiedz:

„Po podpięciu submitowania pętla będzie domykać się automatycznie: wybrane punkty, symulacje, preprocessing, dataset i kolejna iteracja active learning.”
