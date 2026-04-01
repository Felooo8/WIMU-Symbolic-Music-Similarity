# 📚 Analiza literaturowa

## 📋 Tabela analizy literaturowej

| #   | Tytuł / Link                                                                                                                                                                | Autorzy                           | Rok  | **Kod dostępny?**                                                        | **Metryki ewaluacji**                                                                                              | **Zasoby obliczeniowe**             | **Autorski komentarz**                                                                                                                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ---- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation](https://arxiv.org/abs/2412.07948)<br>[GitHub](https://github.com/jryban/frechet-music-distance) | Retkowski, Stępniak, Modrzejewski | 2024 | ✅ [Repo](https://github.com/jryban/frechet-music-distance)              | FMD (Fréchet distance na embeddingach MIDI), FID                                                                   | GPU (MusicBERT/MusicVAE embeddingi) | **Główna referencja projektu.** Definiuje FMD jako baseline. Pokazuje jak metryka działa na datasetach MIDI. Kod gotowy do integracji — kluczowy dla konfrontacji cech statystycznych. |
| 2   | [Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependant](https://arxiv.org/abs/2403.17508)                               | Tailleur et al.                   | 2024 | ❌ (brak oficjalnego repo)                                               | FAD, korelacja Spearmana (FAD vs oceny ludzkie)                                                                    | CPU/GPU (różne modele embeddingowe) | **Metodologia badania odsłuchowego.** Pokazuje, że korelacja Fréchet z oceną ludzką zależy od embeddingu. Uzasadnia nasze badanie perceptualne + Spearman dla ranking cech.            |
| 3   | [MusPy: A Toolkit for Symbolic Music Generation](https://arxiv.org/abs/2008.01951)<br>[GitHub](https://github.com/salu133445/muspy)<br>[Docs](https://muspy.readthedocs.io) | Dong et al.                       | 2020 | ✅ [Repo](https://github.com/salu133445/muspy)                           | pitch_class_entropy, pitch_entropy, pitch_range, scale_consistency, polyphony, empty_beat_rate, groove_consistency | CPU                                 | **Źródło wszystkich cech skalarnych.** Gotowe implementacje w Pythonie — używamy bezpośrednio w `music_features.py`. ISMIR 2020 — wysoki prestiż.                                      |
| 4   | [Learning-Based Methods for Comparing Sequences (Lakh MIDI Dataset)](https://colinraffel.com/publications/thesis.pdf)                                                       | Raffel                            | 2016 | ✅ [Repo](https://colinraffel.com/projects/lmd/)                         | — (dataset paper)                                                                                                  | CPU                                 | **Dokumentacja Lakh MIDI.** Opisuje problemy z "brudnymi" danymi MIDI — uzasadnia wybór oczyszczonej podpróby LMD-matched.                                                             |
| 5   | [Enabling Factorized Piano Music Modeling with the MAESTRO Dataset](https://arxiv.org/abs/1810.12247)                                                                       | Hawthorne et al.                  | 2019 | ✅ [Magenta TensorFlow](https://magenta.tensorflow.org/datasets/maestro) | — (dataset paper)                                                                                                  | CPU                                 | **MAESTRO v3** — wirtuozerska klasyka fortepianowa. Sparowane MIDI+audio. Wbudowane w MusPy — idealne do kontrastu z Lakh MIDI.                                                        |
| 6   | [The NES Music Database: A Multi-Instrumental Dataset](https://arxiv.org/abs/1806.04278)                                                                                    | Donahue et al.                    | 2018 | ✅ [GitHub](https://github.com/chrisdonahue/nesmdb)                      | — (dataset paper)                                                                                                  | CPU                                 | **NES MDB** — muzyka 8-bit. Skrajnie odmienny styl od klasyki — doskonały do testowania robustności cech. Używamy GDrive bypass w ingestion.                                           |

## 📈 Uzasadnienie dalszych prac

### **Dlaczego te prace?**

- **FMD (1)** i **FAD-perception (2)** definiują problem i metodologię
- **MusPy (3)** dostarcza gotowy kod cech — nie wymyślamy koła na nowo
- **Datasety (4,5,6)** zapewniają zróżnicowanie stylistyczne (klasyka vs pop vs chiptune)

### **Postęp implementacyjny (stan na 31.03)**

✅ MusPy metrics z #3 zaimplementowane w music_features.py
✅ 3/4 datasety z #4,5,6 pobrane i sparsowane (ingestion pipeline)
✅ JSD zaimplementowane w similarity/jsd.py — gotowe do konfrontacji z FMD #1
✅ Pipeline end-to-end: make download-data → make run-extraction → make run-similarity

### **Planowane eksperymenty (zgodne z #1,2,3)**

1. **Cechy skalarne** z MusPy (#3) vs FMD (#1)
2. **Histogramy** (pitch class, interwały, durations) vs FMD (#1)
3. **Badanie perceptualne** metodą #2 — 18 par excerptów, skala Likerta
4. **Korelacja Spearman** (#2) — ranking cech po ρ

### **Ryzyka i mitigacja**

| Ryzyko                       | Mitigacja                                       |
| ---------------------------- | ----------------------------------------------- |
| FMD wymaga GPU               | Colab Pro (~10 USD) lub uczelniany klaster      |
| Małe N w badaniu odsłuchowym | Min. 10 słuchaczy + pary kontrolne              |
| Korelacja słaba              | Testujemy 3 miary (JSD, Wasserstein, Euclidean) |

## 🔗 Pełna bibliografia

1. Retkowski et al. (2024). [arXiv:2412.07948](https://arxiv.org/abs/2412.07948)
2. Tailleur et al. (2024). [arXiv:2403.17508](https://arxiv.org/abs/2403.17508)
3. Dong et al. (2020). [arXiv:2008.01951](https://arxiv.org/abs/2008.01951)
4. Raffel (2016). [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/)
5. Hawthorne et al. (2019). [arXiv:1810.12247](https://arxiv.org/abs/1810.12247)
6. Donahue et al. (2018). [arXiv:1806.04278](https://arxiv.org/abs/1806.04278)
