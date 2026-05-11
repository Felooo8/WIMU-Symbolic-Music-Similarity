import json
import logging
import numpy as np
from pathlib import Path

from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class SpearmanAnalysis:
    def __init__(self, similarity_dir: Path, output_dir: Path):
        self.similarity_dir = similarity_dir
        self.output_dir = output_dir

    def run(self) -> dict:
        fmd = self._load_fmd()
        metrics = self._load_metrics()
        metrics.update(self._compute_ensembles(metrics))
        pairs = sorted(fmd.keys())

        logging.info(f"[CORR] Dataset pairs: {pairs}")

        vs_fmd = {}
        for metric_name, metric_values in metrics.items():
            rho, pvalue = self._spearman(metric_values, fmd, pairs)
            vs_fmd[metric_name] = {"rho": rho, "pvalue": pvalue}
            logging.info(f"[CORR] {metric_name:<35} ρ={rho:+.3f}  p={pvalue:.3f}")

        results = {"pairs": pairs, "vs_fmd": vs_fmd}
        self._save(results)
        self._print_ranking(vs_fmd)
        return results

    def _load_fmd(self) -> dict[str, float]:
        raw = _load_json(self.similarity_dir / "fmd_matrix.json")["fmd_matrix"]
        return {
            pair: values["fmd"]
            for pair, values in raw.items()
            if values.get("fmd") is not None
        }

    def _load_metrics(self) -> dict[str, dict[str, float]]:
        metrics = {}

        jsd_path = self.similarity_dir / "jsd_matrix.json"
        if jsd_path.exists():
            jsd_raw = _load_json(jsd_path)
            for dist_type, pairs in jsd_raw.items():
                metrics[f"jsd_{dist_type}"] = pairs

        wass_path = self.similarity_dir / "wasserstein_matrix.json"
        if wass_path.exists():
            wass_raw = _load_json(wass_path)["wasserstein_matrix"]
            for metric_key in ["pitch_class_wasserstein", "interval_wasserstein",
                               "length_note_wasserstein", "average_wasserstein"]:
                metrics[metric_key] = {
                    pair: values[metric_key]
                    for pair, values in wass_raw.items()
                    if metric_key in values
                }

        eucl_path = self.similarity_dir / "euclidean_matrix.json"
        if eucl_path.exists():
            eucl_raw = _load_json(eucl_path)["euclidean_matrix"]
            all_eucl_keys = {k for v in eucl_raw.values() for k in v}
            for key in sorted(all_eucl_keys):
                metrics[key] = {pair: v[key] for pair, v in eucl_raw.items() if key in v}

        mahal_path = self.similarity_dir / "mahalanobis_matrix.json"
        if mahal_path.exists():
            mahal_raw = _load_json(mahal_path)["mahalanobis_matrix"]
            metrics["mahalanobis"] = {pair: v["mahalanobis"] for pair, v in mahal_raw.items()}

        return metrics

    @staticmethod
    def _z_normalize(values: dict[str, float]) -> dict[str, float]:
        arr = np.array(list(values.values()), dtype=float)
        std = arr.std()
        if std == 0:
            return {k: 0.0 for k in values}
        mean = arr.mean()
        return {k: (v - mean) / std for k, v in values.items()}

    def _combine(self, metrics: dict, component_names: list[str]) -> dict[str, float]:
        available = [n for n in component_names if n in metrics]
        if not available:
            return {}
        normalized = {n: self._z_normalize(metrics[n]) for n in available}
        all_pairs = set.intersection(*[set(v.keys()) for v in normalized.values()])
        return {pair: sum(normalized[n][pair] for n in available) for pair in all_pairs}

    def _compute_ensembles(self, metrics: dict) -> dict[str, dict[str, float]]:
        ensembles = {
            "ensemble_intervals": self._combine(metrics, ["jsd_interval", "interval_wasserstein"]),
            "ensemble_interval_mahal": self._combine(metrics, ["jsd_interval", "mahalanobis"]),
            "ensemble_top3": self._combine(metrics, ["jsd_interval", "interval_wasserstein", "mahalanobis"]),
        }
        return {k: v for k, v in ensembles.items() if v}

    @staticmethod
    def _spearman(
        a: dict[str, float], b: dict[str, float], pairs: list[str]
    ) -> tuple[float, float]:
        common = [p for p in pairs if p in a and p in b]
        if len(common) < 2:
            return float("nan"), float("nan")
        result = spearmanr([a[p] for p in common], [b[p] for p in common])
        return float(result.statistic), float(result.pvalue)

    def _save(self, results: dict) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "correlation.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"[CORR] Saved: {output_path}")
        return output_path

    def _print_ranking(self, vs_fmd: dict) -> None:
        print("\n=== Ranking metryk vs FMD (korelacja Spearmana) ===")
        ranked = sorted(
            vs_fmd.items(),
            key=lambda x: abs(x[1]["rho"]) if not np.isnan(x[1]["rho"]) else -1,
            reverse=True,
        )
        for name, vals in ranked:
            rho = vals["rho"]
            pvalue = vals["pvalue"]
            print(f"  {name:<35} ρ={rho:+.3f}  p={pvalue:.3f}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    analysis = SpearmanAnalysis(
        similarity_dir=repo_root / "results" / "similarity",
        output_dir=repo_root / "results" / "analysis",
    )
    analysis.run()


if __name__ == "__main__":
    main()
