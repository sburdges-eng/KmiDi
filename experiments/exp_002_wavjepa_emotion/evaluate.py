"""
Evaluate cached embeddings: k-means (NMI, adjusted Rand), optional linear/MLP head accuracy.
Fixed train/val split for reproducibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Experiment dir on path
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))


def run(
    npz_path: Path,
    config_path: Path,
    encoder_name: str,
    use_classifier: bool = True,
) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    seed = config.get("seed", 42)
    split_ratio = config.get("split_ratio", 0.8)

    data = np.load(npz_path, allow_pickle=True)
    X = data["embeddings"]
    y = data["labels"]
    n_classes = len(np.unique(y))
    le = LabelEncoder()
    y_int = le.fit_transform(y)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_int, train_size=split_ratio, random_state=seed, stratify=y_int
        )
    except ValueError:
        # Too few samples for stratify (e.g. fewer val than classes)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_int, train_size=split_ratio, random_state=seed
        )

    # Clustering on train; evaluate on val by assigning val points to nearest cluster
    kmeans = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    kmeans.fit(X_train)
    train_pred = kmeans.predict(X_train)
    val_pred = kmeans.predict(X_val)

    nmi_train = normalized_mutual_info_score(y_train, train_pred)
    ari_train = adjusted_rand_score(y_train, train_pred)
    nmi_val = normalized_mutual_info_score(y_val, val_pred)
    ari_val = adjusted_rand_score(y_val, val_pred)

    results = {
        "encoder": encoder_name,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_classes": n_classes,
        "nmi_train": float(nmi_train),
        "ari_train": float(ari_train),
        "nmi_val": float(nmi_val),
        "ari_val": float(ari_val),
    }

    if use_classifier:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)
        acc_train = clf.score(X_train, y_train)
        acc_val = clf.score(X_val, y_val)
        results["accuracy_train"] = float(acc_train)
        results["accuracy_val"] = float(acc_val)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path, help="Path to embeddings_<encoder>.npz")
    parser.add_argument("--config", type=Path, default=_EXPERIMENT_DIR / "config.yaml")
    parser.add_argument("--encoder", type=str, default=None, help="Infer from npz filename if not set")
    parser.add_argument("--no-classifier", action="store_true", help="Skip linear classifier")
    args = parser.parse_args()

    encoder = args.encoder
    if encoder is None:
        stem = args.npz_path.stem
        if stem.startswith("embeddings_"):
            encoder = stem.replace("embeddings_", "")
        else:
            encoder = "unknown"
    results = run(args.npz_path, args.config, encoder, use_classifier=not args.no_classifier)

    print("Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    main()
