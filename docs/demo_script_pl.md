# Scenariusz Demo SpinGenix V1

## Cel narracji

Pokazujemy pierwszą wersję konsoli SpinGenix dla modelu dwuparametrycznego `Tx/Tz`. Demo nie ma udawać finalnej jakości modelu po pełnej pętli active learning. Ma pokazać, że mamy spójny przepływ badawczy: kontrolowany dataset, model surrogate, predykcję pola, kontekst fazowy, metryki fizyczne i przygotowany handoff do kolejnych symulacji.

Najbezpieczniejsza rama wypowiedzi:

„To jest działająca konsola badawcza SpinGenix V1. Pokazujemy przepływ i mechanikę systemu, natomiast jakość modelu będzie rosnąć po podpięciu kolejnych automatycznych iteracji symulacyjnych.”

## Agenda na 35-45 minut

- 0-5 min: kontekst problemu i co dokładnie pokazuje demo.
- 5-12 min: dataset, splity i zabezpieczenie holdoutów.
- 12-20 min: checkpoint, zakres normalizatora i live prediction.
- 20-28 min: phase maps jako kontekst parametryczny.
- 28-34 min: rekonstrukcje i interpretacja metryk.
- 34-40 min: active-learning handoff i status HPC/Amumax.
- 40-45 min: podsumowanie, pytania, następne kroki.

Jeśli rozmowa jest krótsza, skróć sekcje `Rekonstrukcje` i `Pytania`, ale zachowaj `Dataset`, `Live prediction`, `Phase context` i `Active learning`.

## Przygotowanie przed wejściem

1. Otwórz platformę w przeglądarce.
2. Zrób twarde odświeżenie strony, żeby nie został stary HTML bez CSS/JS.
3. Zostań na widoku `Demo`.
4. Ustaw device na `CPU`, jeśli CUDA predykcja ma ryzyko kolejki/importu albo niestabilności w sesji demo.
5. Upewnij się, że w panelu bocznym widać `60 samples` i checkpoint z runu `param_surrogate_h100_masked_20260524_201446` albo inny najlepszy dostępny checkpoint.
6. Jeśli predykcja działa wolno, pierwszą część prowadź na gotowych phase maps i rekonstrukcjach, a live prediction kliknij dopiero po omówieniu datasetu.

## 0-5 min: otwarcie

Co kliknąć:

1. Otwórz `Demo`.
2. Nie klikaj jeszcze `Generate Field`.
3. Pokaż ogólny układ: panel parametrów po lewej, live output i phase context po prawej.

Co powiedzieć:

„To jest pierwsza wersja konsoli SpinGenix dla modelu dwuparametrycznego `Tx/Tz`. W tym demo nie twierdzimy jeszcze, że model jest finalny fizycznie. Pokazujemy, że pipeline ma już działającą strukturę: dataset, checkpoint modelu, predykcję pola magnetyzacji, metryki i kontekst active learning.”

„Docelowo ten system ma zamykać pętlę: wybieramy punkty w przestrzeni parametrów, uruchamiamy symulacje mikromagnetyczne, preprocessujemy wyniki, dopisujemy je do datasetu i trenujemy kolejną iterację modelu.”

Czego nie mówić na start:

- nie zaczynaj od problemu z Amumax,
- nie pokazuj tracebacków,
- nie obiecuj, że aktualna mapa fazowa jest ostateczną mapą fizyczną.

## 5-12 min: dataset i splity

Co kliknąć:

1. Pokaż `System Snapshot` na ekranie `Demo`.
2. Przejdź do zakładki `Dataset`.
3. Pokaż liczbę próbek, zakres `Tx/Tz`, splity oraz `Latest Samples`.

Co powiedzieć:

„Dataset startowy ma 60 próbek. To jest bootstrap, czyli kontrolowany punkt startowy do uczenia aktywnego, a nie jeszcze duży finalny dataset.”

„Splity są jawne: 39 próbek train, 9 val, 9 test holdout i 3 boundary holdout. Ważne jest to, że holdouty są zamrożone. Active learning nie może ich użyć jako nowych punktów treningowych ani jako kandydatów akwizycji.”

„To zabezpiecza nas przed leakage. Jeśli po kolejnych iteracjach model poprawia się na holdoutach, możemy traktować to jako bardziej wiarygodny sygnał generalizacji, a nie efekt przypadkowego dopisania tych samych punktów do treningu.”

Dodatkowy komentarz, jeśli ktoś pyta o małą liczbę próbek:

„W tym typie problemu pojedyncza próbka jest droga, bo pochodzi z symulacji mikromagnetycznej. Dlatego zamiast od razu robić gęstą siatkę, budujemy aktywny wybór kolejnych punktów.”

## 12-20 min: checkpoint i live prediction

Co kliknąć:

1. Wróć do `Demo`.
2. W panelu `Model Input` pokaż checkpoint i zakres `Tx/Tz envelope`.
3. Kliknij preset `Interior`.
4. Kliknij `Generate Field`.
5. Po predykcji pokaż obraz komponentowy `Mx/My/Mz` i `Physics Readout`.
6. Kliknij `High Tx` albo `High Tz`, potem ponownie `Generate Field`.

Co powiedzieć:

„Checkpoint niesie ze sobą normalizator parametrów. Platforma pokazuje zakres `Tx/Tz`, na którym checkpoint był uczony, i ostrzega, jeśli wychodzimy poza ten zakres.”

„Dla wybranego punktu model generuje kanoniczne pole magnetyzacji `200x200x3`. W demo pokazujemy komponenty `Mx`, `My` i `Mz`, bo to jest stabilniejsza reprezentacja do rozmowy technicznej niż efektowny obraz HSL.”

„Od razu liczymy proste deskryptory: średnie komponenty magnetyzacji, `MeanMz_abs`, uporządkowanie w płaszczyźnie i przybliżony ładunek topologiczny `Q`. To nie zastępuje pełnej analizy fizycznej, ale daje szybki readout do porównania punktów.”

Jeśli predykcja jest wolna:

„Pierwsze wywołanie ładuje checkpoint i model do procesu serwera, więc może być wolniejsze. Kolejne predykcje powinny być szybsze dzięki cache po stronie backendu.”

Jeśli ktoś pyta o jakość predykcji:

„To jest model z bootstrapu i pojedynczych dotychczasowych runów. Jego rola w tym demo to pokazać działający interfejs i sposób walidacji. Ostateczna jakość będzie zależała od kolejnych iteracji AL i od stabilnego dopływu nowych symulacji.”

## 20-28 min: phase context

Co kliknąć:

1. Na `Demo` pokaż panel `Phase Context`.
2. Przejdź do zakładki `Phase Map`.
3. Pokaż najpierw mapę `Dataset |MeanMz|`.
4. Następnie pokaż `MeanMz_signed` i mapy modelowe, jeśli są dostępne.
5. Wskaż, że punkty źródłowe są widoczne i ważniejsze niż samo tło interpolacji.

Co powiedzieć:

„Phase map nie jest tu traktowana jako finalny wykres publikacyjny. To jest kontekst: gdzie jesteśmy w przestrzeni `Tx/Tz`, gdzie mamy realne symulacje i jakie obszary wyglądają na interesujące.”

„Najważniejsze są punkty źródłowe. Interpolacja jest pomocnicza. Dlatego unikamy zbyt agresywnego smoothingu i ekstrapolowania poza obszarem danych, bo to potrafi tworzyć artefakty sugerujące strukturę, której dataset jeszcze nie uzasadnia.”

„Dla `MeanMz_abs` używamy spokojnej palety sekwencyjnej, bo wartość jest nieujemna. Dla `MeanMz_signed` używamy palety rozbieżnej z centrum w zerze, bo znak ma znaczenie fizyczne.”

Jeśli ktoś mówi, że mapa wygląda zgrubnie:

„Tak, i to jest intencjonalnie uczciwe. Przy 60 próbkach wolę pokazać czytelne punkty i umiarkowane tło niż przesadnie gładką mapę, która wygląda lepiej, ale byłaby mniej wiarygodna.”

## 28-34 min: rekonstrukcje i metryki

Co kliknąć:

1. Przejdź do `Prediction Detail`.
2. Pokaż aktualną predykcję komponentową.
3. Pokaż `Reconstruction Checks`.
4. Jeśli są dostępne obrazy `components`, pokaż je przed HSL.

Co powiedzieć:

„Rekonstrukcje traktujemy diagnostycznie. Komponenty `Mx/My/Mz` pozwalają porównać target, prediction i różnicę w sposób konsekwentny między próbkami.”

„HSL bywa atrakcyjny wizualnie, ale przy prawie jednorodnych stanach może sugerować sztuczne różnice kolorystyczne. Do demo technicznego bezpieczniejsze są komponenty z ustalonym zakresem kolorów.”

„Na tym etapie ważniejsze jest, żeby prezentacja była konsekwentna i audytowalna niż żeby wyglądała jak finalna analiza fizyczna.”

## 34-40 min: active-learning handoff

Co kliknąć:

1. Przejdź do `Active Learning`.
2. Wybierz najnowszy plik `acquisition_iter1.csv`, jeśli nie jest wybrany automatycznie.
3. Pokaż liczbę wybranych punktów oraz zakres `Tx/Tz`.
4. Jeśli trzeba, przejdź na `Runs` i pokaż, że runy, checkpointy, phase maps i acquisition files są wykrywane z `results/`.

Co powiedzieć:

„Dry-run active learning na H100 już działa: trenuje model, liczy uncertainty i wybiera kolejne punkty. Acquisition wyklucza punkty, które są już w datasecie albo registry, i nie dotyka zamrożonych holdoutów.”

„To jest ważna część architektury: model nie jest tylko statyczną predykcją. Ma wskazywać, które nowe symulacje są najbardziej informacyjne dla następnej iteracji.”

„Po podpięciu submitowania pętla będzie działać w schemacie: wybrane punkty, symulacje, preprocessing, aktualizacja datasetu, kolejny trening i kolejna akwizycja.”

## 40-45 min: blocker HPC i zamknięcie

Co powiedzieć o blockerze:

„Sama pętla AL dochodzi do wyboru nowych punktów. Obecny blocker jest integracyjny: na tym HPC dopinamy właściwy sposób uruchamiania Amumax przez kontener, wrapper albo moduł. Walidacja zatrzymuje submit przed `sbatch`, więc to nie jest problem modelu ani selekcji punktów.”

Jeśli ktoś dopytuje technicznie:

„Kod ma już miejsce na konfigurację ścieżki do binarki i modułu CUDA. Nie chcemy jednak zgadywać sposobu uruchamiania na klastrze, bo to musi być zgodne z lokalnym wrapperem/kontenerem. Gdy dostaniemy właściwą komendę, podmieniamy konfigurację submitowania i domykamy pętlę.”

Czego nie rozwijać bez potrzeby:

- brakującej ścieżki do starej binarki,
- pełnych tracebacków,
- detali Slurma, które nie wnoszą nic do oceny modelu,
- obietnic, że pełny AL zakończy się tego samego dnia.

Zamknięcie:

„Podsumowując: mamy działającą konsolę demo, kontrolowany dataset ze splitami, checkpoint z zakresem normalizatora, live prediction, phase context i gotowy mechanizm wyboru kolejnych punktów. Następny krok to poprawne podpięcie submitowania Amumax na tym HPC i wykonanie kolejnych iteracji active learning.”

## Pytania, które mogą paść

### Czy 60 próbek wystarczy?

„Na finalny model raczej nie. Na bootstrap i demonstrację pętli aktywnego uczenia tak. Celem jest właśnie nie marnować symulacji na gęstą siatkę, tylko wybierać kolejne punkty informacyjnie.”

### Czy phase map jest fizycznie wiarygodna?

„Jest wiarygodna jako wizualizacja obecnego datasetu i pomocniczego modelu, ale nie jako finalny diagram fazowy. Punkty źródłowe są twardą informacją; interpolacja jest kontekstem.”

### Czy model umie ekstrapolować?

„Platforma ostrzega przy wyjściu poza envelope checkpointu. Możemy technicznie wygenerować predykcję poza zakresem, ale nie traktujemy jej z takim samym zaufaniem.”

### Dlaczego nie pokazujemy tylko HSL?

„HSL dobrze kondensuje kierunek wektora do jednego obrazu, ale potrafi być mylący przy stanach prawie jednorodnych. Komponenty `Mx/My/Mz` są mniej efektowne, ale stabilniejsze do oceny.”

### Co dokładnie zostało zrobione w active learning?

„Dry-run trenuje model, liczy uncertainty na siatce kandydatów i zapisuje acquisition CSV. Realny submit zatrzymuje się obecnie na walidacji sposobu uruchamiania Amumax na HPC.”

### Co będzie sukcesem następnej iteracji?

„Uruchomienie symulacji z acquisition CSV, preprocessing wyników `.zarr`, dopisanie nowych próbek do datasetu i porównanie metryk modelu na zamrożonych holdoutach.”

## Awaryjna wersja 10-minutowa

1. `Demo`: pokaż `System Snapshot` i powiedz, że to konsola V1 przepływu `Tx/Tz -> field`.
2. `Dataset`: pokaż 60 próbek i zamrożone holdouty.
3. `Demo`: kliknij `Interior` i `Generate Field`.
4. `Phase Map`: pokaż `Dataset |MeanMz|`, podkreśl punkty źródłowe.
5. `Active Learning`: pokaż acquisition CSV.
6. Zamknij: „blocker jest integracyjny po stronie submitowania Amumax, nie po stronie modelu ani AL selection.”

## Jednozdaniowa teza demo

„SpinGenix ma już działający szkielet aktywnego uczenia dla mikromagnetyki: kontrolowany dataset, surrogate model, live prediction, phase context i wybór kolejnych symulacji; brakujący element to dopięcie klastrowego sposobu uruchamiania Amumax.”
