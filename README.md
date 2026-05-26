# 🎵 Cechy statystyczne jako proxy dla miar podobieństwa datasetów muzyki symbolicznej

> **WIMU 2025/2026 — Projekt nr 3 — Zespół nr 6**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/badge/linter-ruff-orange)](https://github.com/astral-sh/ruff)

---

## 📖 O projekcie

Projekt bada, czy **proste cechy statystyczne** ekstrahowane z plików MIDI (rozkład pitch class, histogram interwałów, entropia tonalna, długości nut, polifonia, groove consistency) mogą stanowić **skuteczny i tani proxy** dla zaawansowanych metryk opartych na embeddingach.

**Główne pytanie badawcze:**  
Czy cechy statystyczne są wystarczającym substytutem Frechet Music Distance (FMD) i oceny perceptualnej człowieka — i które z nich korelują z tymi miarami najsilniej? [1, 2]

**Przepływ eksperymentu:**

```
Datasety MIDI
    │
    ▼
Ekstrakcja cech statystycznych (MusPy) [3]
    │
    ├──► Macierze podobieństwa (JSD / Wasserstein / Euclidean)
    │
    ├──► Frechet Music Distance (FMD)
    │
    └──► Badanie odsłuchowe (oceny ludzkie)
              │
              ▼
        Korelacja Spearmana → ranking cech
```

---

## 👥 Autorzy — Zespół nr 6

| Imię i Nazwisko    | GitHub                                         |
| ------------------ | ---------------------------------------------- |
| Feliks Bańka       | [@Felooo8](https://github.com/Felooo8)         |
| Mateusz Baran      | [@matiuszaa](https://github.com/matiuszaa)     |
| Paulina Staszewska | [@paullastasz](https://github.com/paullastasz) |

---

## 📦 Datasety

| Dataset                         | Styl                         | Próba        | Źródło                                                  |
| ------------------------------- | ---------------------------- | ------------ | ------------------------------------------------------- |
| MAESTRO v3                      | Muzyka klasyczna (fortepian) | 1 000 plików | [link](https://magenta.tensorflow.org/datasets/maestro) |
| Lakh MIDI Dataset (LMD-matched) | Pop / rock / mixed           | 1 000 plików | [link](https://colinraffel.com/projects/lmd/)           |
| JSB Chorales                    | Chorały Bacha (4-głosowe)    | ~400 plików  | wbudowany w MusPy [3]                                   |
| NES Music Database              | Muzyka 8-bit / chiptune      | 500 plików   | [link](https://github.com/chrisdonahue/nesmdb)          |

---

## 🛠️ Stack technologiczny

| Warstwa                 | Technologie                                                                |
| ----------------------- | -------------------------------------------------------------------------- |
| Język                   | Python 3.11                                                                |
| Analiza muzyczna        | MusPy, pretty_midi, music21                                                |
| Statystyki              | numpy, scipy                                                               |
| FMD                     | [frechet-music-distance](https://github.com/jryban/frechet-music-distance) |
| Śledzenie eksperymentów | Weights & Biases (W&B)                                                     |
| Wizualizacja            | matplotlib, seaborn                                                        |
| Jakość kodu             | black, ruff, poetry                                                        |
| Testy                   | pytest                                                                     |
| Dokumentacja            | Markdown w katalogu `docs/`                                                |

---

## 🚀 Instalacja i uruchomienie

### Wymagania

- Python 3.11
- [Poetry](https://python-poetry.org/) lub pip
- (Opcjonalnie) GPU dla obliczeń FMD

## Instalacja

### 1. Zależności Python

```bash
poetry install
```

### 2. Zależności systemowe

Do renderowania audio i odsłuchu próbek wymagane są:

- `fluidsynth`
- `ffmpeg`
- soundfont General MIDI, np. `FluidR3_GM.sf2`

Przykład dla Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y fluidsynth ffmpeg fluid-soundfont-gm
```

Po instalacji upewnij się, że soundfont jest dostępny lokalnie, np.:

```bash
ls /usr/share/sounds/sf2/FluidR3_GM.sf2
```

### Uruchomienie krok po kroku

```bash
# 1. Pobierz i przygotuj datasety
make download-data

# 2. Ekstrakcja cech statystycznych
make run-extraction

# 3. Obliczenie macierzy podobieństwa JSD
make run-similarity

# 3b. Obliczenie macierzy podobieństwa Wasserstein (Earth Mover's Distance)
make run-wasserstein

# 3c. Obliczenie FMD
make run-fmd

# 3d. Korelacja rang Spearmana metryk względem FMD
make run-correlation

# 3e. Analiza sensowności datasetów i baseline klasyfikator
make run-baseline

# 4. Uruchomienie testów
make test

# 5. Uruchomienie wszystkich kroków naraz
make all

# 6. Lokalna weryfikacja artefaktów i testów
make verify

# 7. Przygotowanie próbek do badania odsłuchowego
poetry run python listening_study/export_samples.py
```

### Konfiguracja

Wszystkie parametry eksperymentu (ścieżki do danych, rozmiary próbek, model FMD) konfigurowane są w pliku `configs/config.yaml` — oddzielnie od kodu wykonywalnego.

```yaml
# configs/config.yaml (przykład)
datasets:
  maestro:
    path: data/raw/maestro
    sample_size: 1000
  lakh:
    path: data/raw/lakh
    sample_size: 1000
fmd:
  model: musicbert # lub musicvae
  batch_size: 32
wandb:
  project: wimu-proj3
  entity: <team_name>
```

---

## 📁 Struktura projektu

```
.
├── configs/                  # konfiguracja YAML (oddzielona od kodu)
├── data/
│   ├── raw/                  # oryginalne pliki MIDI (niemutowalne)
│   └── processed/            # pliki przetworzone przez MusPy
├── features/
│   ├── scalar.py             # cechy skalarne z MusPy metrics
│   ├── distributions.py      # histogramy pitch class, interwałów, długości nut
│   └── aggregate.py          # agregacja per dataset → DataFrame / CSV
├── similarity/
│   ├── jsd.py                # Jensen-Shannon Divergence
│   ├── wasserstein.py        # Wasserstein / EMD
│   └── euclidean.py          # Euclidean baseline
├── fmd/
│   └── compute_fmd.py        # wrapper na frechet-music-distance
├── listening_study/
│   └── export_samples.py     # eksport próbek MP3 do badania odsłuchowego
├── analysis/
│   ├── correlation.py        # korelacja Spearmana
│   └── visualize.py          # heatmapy, wykresy korelacji
├── tests/                    # pytest
├── docs/
│   ├── literature.md         # tabela analizy literaturowej
│   ├── design_proposal.md    # design proposal projektu
│   └── progress.md           # stan realizacji względem proposalu
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 🎧 Badanie odsłuchowe

Projekt zawiera skrypt `listening_study/export_samples.py`, który przygotowuje próbki audio MP3 do prostego badania odsłuchowego między parami datasetów. Skrypt losuje reprezentatywne pliki z wybranych zbiorów, konwertuje je z formatu MusPy JSON do MIDI, a następnie renderuje do WAV i MP3.

Wygenerowane pliki trafiają do katalogu:

```text
results/listening_pairs/
```

oraz do pliku:

```text
results/listening_pairs/manifest.csv
```

który mapuje próbki odsłuchowe na źródłowe pliki wejściowe.

Uruchomienie:

```bash
poetry run python listening_study/export_samples.py
```

Wymagania systemowe:

- `fluidsynth`
- `ffmpeg`
- soundfont General MIDI, np. `FluidR3_GM.sf2`

---

## 📊 Wyniki (uzupełniane na bieżąco)

Obecna implementacja zapisuje wyniki pośrednie i artefakty wizualne do katalogu `results/`, w szczególności:

- histogramy rozkładów cech dla datasetów,
- plik `results/distributions/distributions.json`,
- plik `results/features/features.json`,
- plik `results/features/summary_stats.json`,
- plik `results/similarity/jsd_matrix.json`,
- plik `results/similarity/wasserstein_matrix.json`,
- plik `results/similarity/fmd_matrix.json`,
- plik `results/analysis/correlation.json`,
- plik `results/analysis/baseline_results.json`,
- wykres `results/analysis/pca_scatter.png`,
- mapa cieplna `results/similarity/heatmap.png`.

Aktualny stan funkcjonalny projektu opisaliśmy także w `docs/progress.md`. Moduł FMD i analiza korelacji Spearmana są zaimplementowane; `make run-correlation` wymaga obecności pliku `results/similarity/fmd_matrix.json`. Dla badania odsłuchowego przygotowano infrastrukturę eksportu próbek; same oceny ludzkie są etapem eksperymentalnym w toku.

### Wasserstein Distance Matrix (Earth Mover's Distance)

| Para datasetów          | pitch_class | interval | length_note |  average  |
| ----------------------- | :---------: | :------: | :---------: | :-------: |
| lakh_midi vs maestro_v3 |    0.641    |  5.309   |    2.063    | **2.671** |
| lakh_midi vs nes_mdb    |    1.702    |  2.480   |    4.653    | **2.945** |
| maestro_v3 vs nes_mdb   |    1.207    |  6.568   |    3.741    | **3.839** |

Wyniki są muzycznie sensowne: para `maestro_v3 vs nes_mdb` wykazuje największy
dystans (3.839 average), co odzwierciedla dużą różnicę między klasyczną muzyką
fortepianową a chiptune, czyli muzyką 8-bitową. Najsilniej widać to w rozkładzie
interwałów (`interval = 6.568`), gdzie oba datasety różnią się najbardziej.

Para `lakh_midi vs maestro_v3` ma najniższy średni dystans (2.671), co sugeruje,
że szeroki, wielogatunkowy zbiór Lakh MIDI zawiera materiał bliższy klasycznym
strukturom wysokościowym niż NES-MDB. Z kolei `lakh_midi vs nes_mdb` mocniej
różni się w rozkładzie długości nut (`length_note = 4.653`), czyli w profilu
rytmicznym / czasowym.

Wasserstein distance, w odróżnieniu od JSD, uwzględnia metrykę osi X histogramu:
przesunięcie masy o jeden bin jest traktowane jako mniejsza różnica niż
przesunięcie o wiele binów. Dzięki temu metryka lepiej oddaje intuicję, że
podobne interwały albo długości nut powinny być bliższe niż wartości odległe.

### Zestawienie wyników końcowych

Kolumna JSD pokazuje średnią arytmetyczną z wartości `pitch_class` i `interval`
zapisanych w `results/similarity/jsd_matrix.json`: `JSD = (JSD_pitch_class + JSD_interval) / 2`.

| Para datasetów                    | JSD | Wasserstein |      FMD | Średnia ocena słuchaczy | Uwagi                       |
| --------------------------------- | --: | ----------: | -------: | ----------------------: | --------------------------- |
| maestro_v3 vs music_net           | 0.0664 |     18.7364 | 320.8859 |                     4.0 | porównanie zbliżonych domen |
| maestro_v3 vs jsb_chorales        | 0.2165 |     54.3375 | 669.3721 |                     4.0 | klasyczna vs chorały        |
| nes_mdb vs maestro_v3             | 0.0940 |   2220.4330 | 537.0342 |                     1.5 | chiptune vs fortepian       |
| nes_mdb vs jsb_chorales           | 0.0708 |   2270.0531 | 668.5928 |                     1.5 | syntetyczne vs barok        |
| lakh_midi_rock vs lakh_midi_metal | 0.0167 |     18.4345 | 130.2720 |                     3.2 | bliskie stylistycznie       |
| lakh_midi_pop vs lakh_midi_rock   | 0.0092 |     16.4441 |  61.0268 |                     2.7 | popularne gatunki           |

Wartości w kolumnie oceny słuchaczy pochodzą z 6 odpowiedzi w badaniu ankietowym: https://tally.so/r/ODbP0g.

### Badanie odsłuchowe

Dla każdej pary datasetów przygotowano próbki audio eksportowane skryptem `listening_study/export_samples.py`. Odpowiedzi z ankiety odsłuchowej (https://tally.so/r/ODbP0g) zostały zagregowane do wspólnej skali podobieństwa i zestawione z wynikami JSD, Wasserstein Distance oraz FMD.

### Wstępna interpretacja

Wyniki są zgodne z odsłuchem: pary klasyczne (`maestro_v3` vs `music_net` oraz
`maestro_v3` vs `jsb_chorales`) mają najwyższe oceny słuchaczy, a pary z
`nes_mdb` najniższe. Najsilniejszą zgodność z FMD w analizie Spearmana uzyskały
metryki oparte na interwałach: `jsd_interval`, `interval_wasserstein` i
`average_wasserstein` osiągnęły ρ = 0.829 przy p = 0.042. Z kolei
`jsd_pitch_class` osiągnął ρ = 0.600 przy p = 0.208, więc sam rozkład klas
wysokości nie daje istotnej korelacji na tej próbie.

Korelacja Spearmana została policzona dla 6 par datasetów, dlatego należy
traktować ją jako wynik ilustracyjny i sanity check, a nie mocny dowód
statystyczny. Przy tak małym `n` pojedyncza para odstająca może istotnie zmienić
wartości ρ i p-value.

---

## Prezentacja wizualna projektu

W ramach projektu stworzono wideo, które pokazuje działanie projektu w skrócie, przykładowe próbki audio użyte do badań odsłuchowych oraz histogramy.

[Klikni tutaj, by przenieść się do pliku wideo](https://github.com/Felooo8/WIMU-Symbolic-Music-Similarity/blob/main/video.mp4).

---

## 📚 Referencje

1. Retkowski, Stępniak, Modrzejewski (2024). _Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation_. [arXiv:2412.07948](https://arxiv.org/abs/2412.07948)
2. Manor, Leibovich (2024). _Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependant_. [arXiv:2403.17508](https://arxiv.org/abs/2403.17508)
3. Dong et al. (2020). _MusPy: A Toolkit for Symbolic Music Generation_. ISMIR 2020. [GitHub](https://github.com/salu133445/muspy)

---

## 📄 Licencja

MIT License — szczegóły w pliku [LICENSE](LICENSE).
