import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

STYLE_MAPS = {
    "coarse": {
        "maestro_v3": "classical",
        "music_net": "classical",
        "jsb_chorales": "classical",
        "nes_mdb": "chiptune",
        "lakh_midi_rock": "rock",
        "lakh_midi_metal": "metal",
        "lakh_midi_pop": "pop",
        "lakh_midi_jazz": "jazz",
        "lakh_midi_electronic": "electronic",
    },
    "with_chorale": {
        "maestro_v3": "classical",
        "music_net": "classical",
        "jsb_chorales": "chorale",
        "nes_mdb": "chiptune",
        "lakh_midi_rock": "rock",
        "lakh_midi_metal": "metal",
        "lakh_midi_pop": "pop",
        "lakh_midi_jazz": "jazz",
        "lakh_midi_electronic": "electronic",
    },
}


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


def _build_matrix(features_by_dataset: dict, style_map: dict[str, str]) -> tuple:
    rows = []
    labels = []
    dropped_nan = defaultdict(int)
    kept_by_style = defaultdict(int)
    excluded_datasets = {}

    for dataset_name, dataset_rows in sorted(features_by_dataset.items()):
        style = style_map.get(dataset_name)
        if style is None:
            excluded_datasets[dataset_name] = len(dataset_rows)
            logging.info("[GENRE] %s: excluded from genre mapping", dataset_name)
            continue

        for row in dataset_rows:
            if not _is_valid_row(row):
                dropped_nan[dataset_name] += 1
                continue
            rows.append([float(row[feature]) for feature in FEATURE_COLUMNS])
            labels.append(style)
            kept_by_style[style] += 1

        logging.info(
            "[GENRE] %s -> %s: kept=%d dropped_nan=%d",
            dataset_name,
            style,
            len(dataset_rows) - dropped_nan[dataset_name],
            dropped_nan[dataset_name],
        )

    if not rows:
        raise ValueError("No valid feature rows found after applying genre mapping.")

    return (
        np.asarray(rows, dtype=float),
        np.asarray(labels),
        dict(sorted(kept_by_style.items())),
        dict(sorted(dropped_nan.items())),
        excluded_datasets,
    )


def _models() -> dict:
    return {
        "knn_k3": KNeighborsClassifier(n_neighbors=3),
        "svm_rbf": SVC(kernel="rbf", C=10.0, gamma="scale"),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def _cross_val_scores(model, x_scaled: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x_scaled, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def _cross_val_predictions(model, x_scaled: np.ndarray, y: np.ndarray) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, x_scaled, y, cv=cv)


def _save_confusion_matrix(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("[GENRE] Saved confusion matrix: %s", output_path)


def _evaluate_variant(name: str, features_by_dataset: dict, analysis_dir: Path) -> dict:
    logging.info("[GENRE] Running variant: %s", name)
    style_map = STYLE_MAPS[name]
    x, y, class_counts, dropped_nan, excluded = _build_matrix(features_by_dataset, style_map)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model_results = {}
    fitted_models = _models()
    for model_name, model in fitted_models.items():
        mean, std = _cross_val_scores(model, x_scaled, y)
        model_results[model_name] = {"mean": mean, "std": std}
        print(f"{name} | {model_name}: {mean:.3f} ± {std:.3f}")

    best_model_name = max(model_results, key=lambda item: model_results[item]["mean"])
    y_pred = _cross_val_predictions(fitted_models[best_model_name], x_scaled, y)
    labels = sorted(set(y))
    confusion_filename = f"genre_confusion_matrix_{name}.png"
    confusion_output = analysis_dir / confusion_filename
    _save_confusion_matrix(
        y_true=y,
        y_pred=y_pred,
        labels=labels,
        title=f"Genre classification confusion matrix — {name} ({best_model_name})",
        output_path=confusion_output,
    )

    matrix = confusion_matrix(y, y_pred, labels=labels)
    per_class_accuracy = {
        label: float(matrix[index, index] / matrix[index].sum()) if matrix[index].sum() else 0.0
        for index, label in enumerate(labels)
    }

    return {
        "style_map": style_map,
        "class_counts": class_counts,
        "dropped_nan": dropped_nan,
        "excluded_datasets": excluded,
        "models": model_results,
        "best_model": best_model_name,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
            "image": f"results/analysis/{confusion_filename}",
        },
        "per_class_accuracy": per_class_accuracy,
    }


def _save_results(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("[GENRE] Saved numeric results: %s", output_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    features_path = repo_root / "results" / "features" / "features.json"
    analysis_dir = repo_root / "results" / "analysis"

    logging.info("[GENRE] Loading features: %s", features_path)
    features_by_dataset = _load_json(features_path)

    results = {
        "feature_columns": FEATURE_COLUMNS,
        "variants": {},
        "notes": (
            "JSD/Wasserstein/FMD are dataset-level distances, so this classifier uses "
            "per-file statistical features instead of injecting dataset-level metrics "
            "directly into SVM/KNN."
        ),
    }
    for variant_name in STYLE_MAPS:
        results["variants"][variant_name] = _evaluate_variant(
            variant_name,
            features_by_dataset,
            analysis_dir,
        )

    _save_results(results, analysis_dir / "genre_results.json")


if __name__ == "__main__":
    main()
