import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


FEATURE_COLUMNS = [
    "pitch_class_entropy",
    "pitch_entropy",
    "pitch_range",
    "scale_consistency",
    "polyphony",
    "empty_beat_rate",
    "groove_consistency",
]

SUMMARY_COLUMNS = [
    "polyphony",
    "pitch_entropy",
    "pitch_range",
    "groove_consistency",
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}. Run feature extraction first with "
            "`make run-extraction`."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_valid_row(row: dict) -> bool:
    for feature in FEATURE_COLUMNS:
        value = row.get(feature)
        if value is None:
            return False
        try:
            if np.isnan(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _build_matrix(features_by_dataset: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = []
    labels = []
    dropped = defaultdict(int)
    kept_rows = defaultdict(list)

    for dataset_name, dataset_rows in sorted(features_by_dataset.items()):
        for row in dataset_rows:
            if not _is_valid_row(row):
                dropped[dataset_name] += 1
                continue

            feature_values = [float(row[feature]) for feature in FEATURE_COLUMNS]
            rows.append(feature_values)
            labels.append(dataset_name)
            kept_rows[dataset_name].append(row)

        logging.info(
            "[BASELINE] %s: kept=%d dropped_nan=%d",
            dataset_name,
            len(kept_rows[dataset_name]),
            dropped[dataset_name],
        )

    if not rows:
        raise ValueError("No valid feature rows found after dropping NaN values.")

    return np.asarray(rows, dtype=float), np.asarray(labels), kept_rows


def _dataset_summary(kept_rows: dict) -> dict:
    summary = {}
    for dataset_name, rows in sorted(kept_rows.items()):
        dataset_summary = {"n_files": len(rows)}
        for feature in SUMMARY_COLUMNS:
            values = [float(row[feature]) for row in rows]
            dataset_summary[f"mean_{feature}"] = float(np.mean(values)) if values else float("nan")
        summary[dataset_name] = dataset_summary
    return summary


def _print_summary(summary: dict) -> None:
    print("\n=== Dataset summary ===")
    print(
        f"{'dataset':<24} {'n_files':>7} {'mean_polyphony':>16} "
        f"{'mean_pitch_entropy':>20} {'mean_pitch_range':>18} "
        f"{'mean_groove_consistency':>25}"
    )
    for dataset_name, values in summary.items():
        print(
            f"{dataset_name:<24} {values['n_files']:>7d} "
            f"{values['mean_polyphony']:>16.3f} "
            f"{values['mean_pitch_entropy']:>20.3f} "
            f"{values['mean_pitch_range']:>18.3f} "
            f"{values['mean_groove_consistency']:>25.3f}"
        )


def _save_pca_plot(x_scaled: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    logging.info("[BASELINE] Running PCA")
    pca_points = PCA(n_components=2, random_state=42).fit_transform(x_scaled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    for dataset_name in sorted(set(y)):
        mask = y == dataset_name
        plt.scatter(
            pca_points[mask, 0],
            pca_points[mask, 1],
            label=dataset_name,
            s=28,
            alpha=0.75,
        )
    plt.title("PCA — Statistical Features per Dataset")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info("[BASELINE] Saved PCA scatter plot: %s", output_path)


def _cross_val_scores(model, x_scaled: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x_scaled, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def _top_rf_importances(x_scaled: np.ndarray, y: np.ndarray) -> dict:
    # Fit a separate RF on the full dataset only to describe feature importances.
    # Cross-validation accuracy is computed independently in _cross_val_scores().
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_scaled, y)
    importances = {
        feature: float(importance)
        for feature, importance in zip(FEATURE_COLUMNS, rf.feature_importances_)
    }
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def _save_results(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("[BASELINE] Saved numeric results: %s", output_path)


def _print_feature_importances(importances: dict) -> None:
    print("\n=== Random Forest feature importances (top 5) ===")
    for feature, importance in list(importances.items())[:5]:
        print(f"{feature:<24} {importance:.4f}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    features_path = repo_root / "results" / "features" / "features.json"
    analysis_dir = repo_root / "results" / "analysis"

    logging.info("[BASELINE] Loading features: %s", features_path)
    features_by_dataset = _load_json(features_path)

    logging.info("[BASELINE] Building feature matrix")
    x, y, kept_rows = _build_matrix(features_by_dataset)
    logging.info("[BASELINE] Feature matrix shape: %s", x.shape)

    summary = _dataset_summary(kept_rows)
    _print_summary(summary)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    _save_pca_plot(x_scaled, y, analysis_dir / "pca_scatter.png")

    logging.info("[BASELINE] Running KNN 5-fold cross-validation")
    knn_mean, knn_std = _cross_val_scores(KNeighborsClassifier(n_neighbors=3), x_scaled, y)
    print(f"\nKNN (k=3) accuracy: {knn_mean:.3f} ± {knn_std:.3f}")

    logging.info("[BASELINE] Running Random Forest 5-fold cross-validation")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_mean, rf_std = _cross_val_scores(rf_model, x_scaled, y)
    print(f"Random Forest accuracy: {rf_mean:.3f} ± {rf_std:.3f}")

    importances = _top_rf_importances(x_scaled, y)
    _print_feature_importances(importances)

    results = {
        "dataset_summary": summary,
        "knn_accuracy": {"mean": knn_mean, "std": knn_std},
        "rf_accuracy": {"mean": rf_mean, "std": rf_std},
        "rf_feature_importances": importances,
    }
    _save_results(results, analysis_dir / "baseline_results.json")


if __name__ == "__main__":
    main()
