# Design Proposal: Cechy statystyczne jako proxy dla miar podobieństwa datasetów muzyki symbolicznej

**Kurs:** WIMU 2025/2026  
**Temat nr 3**  
**Zespół:** Feliks Bańka, Mateusz Baran, Paulina Staszewska  
**Repozytorium:** https://github.com/Felooo8/WIMU-Symbolic-Music-Similarity  
**Data:** 18.03.2026  

---

## 1. Cel projektu i pytanie badawcze

Głównym pytaniem badawczym jest: **czy proste cechy statystyczne ekstrahowane z plików MIDI są wystarczającym i tanim proxy dla metryk opartych na embeddingach (Frechet Music Distance) oraz oceny perceptualnej człowieka?**

Projekt ma wymiar zarówno eksperymentalny, jak i aplikacyjny: zbudujemy pipeline, który dla dowolnej pary datasetów muzyki symbolicznej oblicza macierz podobieństwa opartą na cechach statystycznych, a następnie porówna ją z wynikami FMD i badania odsłuchowego przy użyciu korelacji rang Spearmana.

---

## 2. Planowane datasety

Wybrano 4 datasety zapewniające maksymalne zróżnicowanie stylistyczne przy dostępności przez MusPy lub standardowe repozytoria:

| Dataset                             | Styl                        | Planowana próba            | Uwagi                                                   |
|-------------------------------------|-----------------------------|----------------------------|---------------------------------------------------------|
| **MAESTRO v3**                      | Muzyka klasyczna, fortepian | 1 000 plików               | Wirtuozerska, sparowane MIDI + audio, wbudowana w MusPy |
| **Lakh MIDI Dataset (LMD-matched)** | Pop/rock/różne gatunki      | 1 000 losowych plików      | Bardzo „brudny", duże zróżnicowanie stylów              |
| **JSB Chorales**                    | Barokowe chorały (Bach)     | ~400 plików (cały dataset) | Czysty, 4-głosowy, dostępny przez MusPy                 |
| **NES Music Database**              | Muzyka 8-bit / chiptune     | 500 plików                 | Skrajnie odmienny styl, ograniczona polifonia           |

Daje to **6 par** datasetów — macierze podobieństwa 4×4.

---

## 3. Planowane cechy statystyczne

### 3.1 Cechy skalarne (via MusPy metrics)

| Cecha | Opis |
|---|---|
| `pitch_class_entropy` | Entropia rozkładu 12 klas wysokości dźwięku |
| `pitch_entropy` | Entropia rozkładu 128 nut MIDI |
| `pitch_range` | Zakres wysokości (max pitch − min pitch) |
| `scale_consistency` | Udział nut zgodnych z dominującą skalą |
| `polyphony` | Średnia liczba jednocześnie granych nut |
| `empty_beat_rate` | Udział pustych beatów (miara gęstości rytmicznej) |
| `groove_consistency` | Dystanse Hamminga między kolejnymi metrykami rytmicznymi |

### 3.2 Rozkłady (obliczane ręcznie, agregowane per dataset)

| Rozkład | Wymiarowość | Metoda agregacji |
|---|---|---|
| Histogram pitch class | 12-dim | Suma po wszystkich nutach w datasecie |
| Histogram interwałów | 49-dim (±24 półtony) | Suma po kolejnych parach nut |
| Histogram długości nut | 16-dim (log-binnowanie) | Suma po wszystkich nutach |

---

## 4. Planowane miary podobieństwa

Dla każdej pary datasetów i każdego rozkładu obliczamy:

- **Jensen-Shannon Divergence (JSD)** — dla rozkładów dyskretnych (pitch class, interwały); symetryczna, bounded w [0, 1]
- **Wasserstein distance (EMD)** — dla długości nut; zachowuje porządek metryczny
- **Odległość euklidesowa na znormalizowanych wektorach cech skalarnych** — prosty baseline

---

## 5. Frechet Music Distance (FMD)

FMD obliczamy za pomocą oficjalnego repozytorium: https://github.com/jryban/frechet-music-distance

FMD mierzy odległość Frécheta między rozkładami Gaussa dopasowanymi do embeddingów dwóch zbiorów plików MIDI. Model embeddingowy do ustalenia podczas analizy literaturowej (kandydaci: MusicBERT, MusicVAE). Obliczenia FMD dla 6 par datasetów × ~1 000 plików szacujemy na **4–8 h GPU** (Google Colab Pro lub klaster uczelniany).

---

## 6. Badanie odsłuchowe

Metodologia wzorowana na Manor & Leibovich (2024):

- **Słuchacze:** minimum 10 osób (ochotnicy ze środowiska akademickiego)
- **Próbki:** 18 par 15-sekundowych excerptów — 3 pary na każdą z 6 kombinacji datasetów, w tym 3 pary kontrolne wewnątrz-datasetowe (oczekiwana wysoka podobność)
- **Skala:** Likerta 1–5 (1 = zupełnie różna muzyka, 5 = bardzo podobna)
- **Narzędzie:** PQ Toolkit lub Google Forms
- **Zbiórka danych:** 17.04 – 01.05.2026

---

## 7. Analiza korelacji

Dla każdej cechy/miary obliczamy **korelację rang Spearmana** (ρ) między:

1. Wektor podobieństw statystycznych (6 wartości) vs wektor FMD (6 wartości)
2. Wektor podobieństw statystycznych (6 wartości) vs mediana ocen odsłuchowych (6 wartości)
3. FMD vs mediana ocen odsłuchowych — weryfikacja referencji z Manor & Leibovich

Wynik: ranking cech według ρ → bezpośrednia odpowiedź na pytanie badawcze.

---

## 8. Harmonogram

| Termin | Zakres prac |
|---|---|
| **do 18.03.2026** | ✅ Design proposal (ten dokument); setup repo: cookiecutter, venv/poetry, black, flake8, Makefile |
| **18.03 – 25.03** | Analiza literaturowa → tabela w `docs/literature.md`; wczytywanie datasetów przez MusPy; testy parsowania MIDI; konfiguracja W&B |
| **25.03 – 01.04** | Implementacja modułu `features/`: ekstrakcja cech skalarnych i histogramów; pierwsze wizualizacje rozkładów pitch class per dataset |
| **01.04.2026 (śr 23:59)** | ✅ Zgłoszenie gotowości prototypu: działający pipeline ekstrakcja → macierz JSD dla 4 datasetów |
| **01.04 – 10.04** | Pełne macierze podobieństwa (JSD, Wasserstein, Euclidean) dla 6 par × wszystkich cech; demo kodu |
| **10.04.2026 (pt)** | ✅ Spotkanie prototypowe z prowadzącym |
| **10.04 – 17.04** | Obliczenie FMD dla 6 par (GPU); konfiguracja środowiska embeddingowego |
| **17.04 – 01.05** | Projektowanie i przeprowadzenie badania odsłuchowego; zbiórka ocen od słuchaczy |
| **01.05 – 12.05** | Obliczenie korelacji Spearmana; wizualizacje: heatmapy macierzy podobieństwa, wykresy ρ per cecha |
| **12.05 – 25.05** | Draft artykułu: wprowadzenie, metodologia, wyniki, dyskusja; finalizacja kodu; testy; dokumentacja; filmik 3–5 min |
| **25.05.2026 (pon 23:59)** | ✅ Zwolnienie projektu |
| **25.05 – 08.06** | Poprawki; przygotowanie prezentacji publicznej na ostatnim wykładzie |
| **08.06.2026 (pon 23:59)** | ✅ Finalny deadline projektu |

---

## 9. Planowana funkcjonalność i struktura projektu

```
project/
├── data/
│   ├── raw/                  # oryginalne pliki MIDI (niemutowalne)
│   └── processed/            # wczytane przez MusPy, oczyszczone
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
│   └── sample_excerpts.py    # generowanie excerptów do badania odsłuchowego
├── analysis/
│   ├── correlation.py        # korelacja Spearmana: cechy vs FMD vs ludzie
│   └── visualize.py          # heatmapy, wykresy korelacji
├── tests/                    # pytest, min. Python 3.10 i 3.11
├── docs/
│   ├── literature.md         # tabela analizy literaturowej
│   └── usage.md              # instrukcja użytkowania
├── configs/                  # konfiguracja YAML oddzielona od kodu
├── Makefile                  # make install / make test / make run / make fmd
└── README.md
```

---

## 10. Stack technologiczny

| Warstwa | Technologie |
|---|---|
| Język | Python 3.11 |
| Muzyka / MIDI | MusPy, pretty_midi, music21 |
| Cechy statystyczne | numpy, scipy (stats, spatial.distance) |
| FMD | `frechet-music-distance` (github.com/jryban/frechet-music-distance) |
| Śledzenie eksperymentów | Weights & Biases (W&B) |
| Wizualizacja | matplotlib, seaborn |
| Jakość kodu | black, flake8 / ruff, poetry |
| Testy automatyczne | pytest, tox (Python 3.10 + 3.11) |
| Dokumentacja | mkdocs |
| Struktura projektu | cookiecutter data science |
| Konteneryzacja (opcjonalna) | Docker |

---

## 11. Zasoby obliczeniowe

| Zadanie | Szacowany czas | Sprzęt |
|---|---|---|
| Ekstrakcja cech statystycznych (≈4 000 plików MIDI) | ~30 min | CPU |
| Obliczenie FMD dla 6 par × embeddingi | 4–8 h | GPU |
| Badanie odsłuchowe | 2 tygodnie | — |
| Analiza korelacji + wizualizacje | < 5 min | CPU |

Dostęp do GPU: Google Colab Pro (jednorazowy koszt ~10 USD) lub klaster HPC uczelni. Obliczenia GPU planowane w tygodniu **10.04 – 17.04**.

---

## 12. Analiza literaturowa (tabela robocza)

| Praca | Autorzy | Rok | Kod | Modele / Metryki | Zasoby oblicz. | Komentarz |
|---|---|---|---|---|---|---|
| Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation | Retkowski, Stępniak, Modrzejewski | 2024 | ✅ github.com/jryban/frechet-music-distance | FMD (Fréchet na embeddingach MIDI) | GPU | Główna referencja; definiuje FMD; pokazuje użycie na datasetach MIDI |
| Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependant | Manor, Leibovich | 2024 | ❌ | FAD, korelacja Spearmana | CPU/GPU | Kluczowa metodologia badania odsłuchowego; pokazuje zależność korelacji od modelu embeddingowego |
| MusPy: A Toolkit for Symbolic Music Generation | Dong et al. | 2020 | ✅ github.com/salu133445/muspy | pitch_class_entropy, polyphony, groove_consistency, scale_consistency, pitch_range, empty_beat_rate | CPU | Gotowe implementacje wszystkich planowanych cech skalarnych |
| Learning-Based Methods for Comparing Sequences (Lakh MIDI) | Raffel | 2016 | ✅ colinraffel.com/projects/lmd | — | CPU | Opis i dokumentacja datasetu Lakh MIDI |
| Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset | Hawthorne et al. | 2019 | ✅ magenta.tensorflow.org | — | CPU | Opis datasetu MAESTRO |

---

## 13. Bibliografia

1. Retkowski, M., Stępniak, J., Modrzejewski, M. (2024). *Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation*. arXiv:2412.07948. https://arxiv.org/abs/2412.07948  
2. Manor, O., Leibovich, T. (2024). *Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependant*. arXiv:2403.17508. https://arxiv.org/abs/2403.17508  
3. Dong, H. W., et al. (2020). *MusPy: A Toolkit for Symbolic Music Generation*. ISMIR 2020. https://github.com/salu133445/muspy  
4. Raffel, C. (2016). *Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching*. PhD thesis. https://colinraffel.com/projects/lmd/  
5. Hawthorne, C., et al. (2019). *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset*. ICLR 2019. https://magenta.tensorflow.org/datasets/maestro  
