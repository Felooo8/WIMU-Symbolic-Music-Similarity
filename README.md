# 🎵 Cechy statystyczne jako proxy dla miar podobieństwa datasetów muzyki symbolicznej

> **WIMU 2025/2026 — Projekt nr 3 — Zespół nr 6**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/badge/linter-ruff-orange)](https://github.com/astral-sh/ruff)

---

## 📖 O projekcie

Projekt bada, czy **proste cechy statystyczne** ekstrahowane z plików MIDI (rozkład pitch class, histogram interwałów, entropia tonalna, długości nut, polifonia, groove consistency) mogą stanowić **skuteczny i tani proxy** dla zaawansowanych metryk opartych na embeddingach.

**Główne pytanie badawcze:**  
Czy cechy statystyczne są wystarczającym substytutem Frechet Music Distance (FMD) i oceny perceptualnej człowieka — i które z nich korelują z tymi miarami najsilniej?

**Przepływ eksperymentu:**

```
Datasety MIDI
    │
    ▼
Ekstrakcja cech statystycznych (MusPy)
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

| Imię i Nazwisko    | GitHub                                 |
| ------------------ | -------------------------------------- |
| Feliks Bańka       | [@Felooo8](https://github.com/Felooo8) |
| Mateusz Baran      | [@...](https://github.com)             |
| Paulina Staszewska | [@...](https://github.com)             |

---

## 📦 Datasety

| Dataset                         | Styl                         | Próba        | Źródło                                                  |
| ------------------------------- | ---------------------------- | ------------ | ------------------------------------------------------- |
| MAESTRO v3                      | Muzyka klasyczna (fortepian) | 1 000 plików | [link](https://magenta.tensorflow.org/datasets/maestro) |
| Lakh MIDI Dataset (LMD-matched) | Pop / rock / mixed           | 1 000 plików | [link](https://colinraffel.com/projects/lmd/)           |
| JSB Chorales                    | Chorały Bacha (4-głosowe)    | ~400 plików  | wbudowany w MusPy                                       |
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
| Testy                   | pytest, tox                                                                |
| Dokumentacja            | mkdocs                                                                     |

---

## 🚀 Instalacja i uruchomienie

### Wymagania

- Python 3.11
- [Poetry](https://python-poetry.org/) lub pip
- (Opcjonalnie) GPU dla obliczeń FMD

### Instalacja

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>

# Instalacja zależności przez Poetry
make install

# lub ręcznie przez pip
pip install -r requirements.txt
```

### Uruchomienie krok po kroku

```bash
# 1. Pobierz i przygotuj datasety
make download-data

# 2. Ekstrakcja cech statystycznych
make run-extraction

# 3. Obliczenie macierzy podobieństwa (JSD, Wasserstein, Euclidean)
make run-similarity

# 4. Obliczenie FMD (wymaga GPU)
make run-fmd

# 5. Analiza korelacji i wizualizacje
make run-analysis

# 6. Uruchomienie testów
make test

# 7. Uruchomienie wszystkich kroków naraz
make all
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
│   └── sample_excerpts.py    # generowanie excerptów do badania odsłuchowego
├── analysis/
│   ├── correlation.py        # korelacja Spearmana
│   └── visualize.py          # heatmapy, wykresy korelacji
├── tests/                    # pytest
├── docs/
│   ├── literature.md         # tabela analizy literaturowej
│   └── usage.md              # szczegółowa instrukcja użytkowania
├── design_proposal.md        # design proposal projektu
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 📊 Wyniki (uzupełniane na bieżąco)

Wyniki eksperymentów (macierze podobieństwa, wartości FMD, korelacje Spearmana) publikowane są w katalogu `results/` oraz śledzone na W&B:  
🔗 [Dashboard W&B](https://wandb.ai/<team_name>/wimu-proj3) _(link aktywny po uruchomieniu eksperymentów)_

---

## 📚 Referencje

1. Retkowski, Stępniak, Modrzejewski (2024). _Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation_. [arXiv:2412.07948](https://arxiv.org/abs/2412.07948)
2. Manor, Leibovich (2024). _Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependant_. [arXiv:2403.17508](https://arxiv.org/abs/2403.17508)
3. Dong et al. (2020). _MusPy: A Toolkit for Symbolic Music Generation_. ISMIR 2020. [GitHub](https://github.com/salu133445/muspy)

---

## 📄 Licencja

MIT License — szczegóły w pliku [LICENSE](LICENSE).
