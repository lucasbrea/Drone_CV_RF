"""
Random Forest v8 — standalone, reproducible trainer.

Trains the v8 pasture-species Random Forest on the bundled training database and
reports an HONEST grouped cross-validated comparison (vs a minimum-distance-to-
means baseline). Self-contained: needs only training_data.db, no drone imagery.

What's different from v2:
  * 15 features per patch — 9 spectral (5 indices + 4 bands) PLUS 6 neighborhood
    statistics (std/mean of NDVI/NIR/GRI over 3x3 / 5x5 patch windows).
  * 5 classes — Water Hole is folded into Bare Soil (both are non-vegetated
    ground); the database keeps them separate so the merge stays explicit here.
  * Grouped cross-validation (StratifiedGroupKFold on group_id) — all patches
    sampled around one placemark / MRK point stay in the same fold, so the score
    is not inflated by near-duplicate neighbouring patches leaking across folds.
    This is the honest number (~72% vs the ~optimistic non-grouped figure).

Each row in training_data.db is one ~30 cm ground-truth patch: class label, its
centre lat/lon, a group_id, and the 15 features.

Run:
    pip install -r requirements.txt
    python train_rf_v8.py

Outputs (written next to this script):
    confusion_matrices.png   RF vs MDM, grouped CV
    feature_importance.png   which features drive the Random Forest
    cv_metrics.csv           per-class precision / recall / F1
    rf_v8_model.pkl          the trained Random Forest (+ feature/class metadata)
"""
import argparse
import csv
import gzip
import os
import pickle
import sqlite3
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt

FEATURES = ['ndvi', 'ndre', 'gndvi', 'gri', 'sr', 'red', 'green', 'nir', 'rededge',
            'ndvi_std3', 'nir_std3', 'gri_std3', 'ndvi_mean5', 'nir_mean5', 'ndvi_std5']
FEATURE_LABELS = ['NDVI', 'NDRE', 'GNDVI', 'GRI', 'SR', 'Red', 'Green', 'NIR', 'RedEdge',
                  'NDVI_std3', 'NIR_std3', 'GRI_std3', 'NDVI_mean5', 'NIR_mean5', 'NDVI_std5']
# Water Hole -> Bare Soil (both non-vegetated ground).
MERGE = {'Water Hole': 'Bare Soil'}
# v8 tuned Random Forest settings.
RF_PARAMS = dict(n_estimators=500, max_depth=30, min_samples_split=5,
                 min_samples_leaf=2, max_features='log2')
HERE = os.path.dirname(os.path.abspath(__file__))


def load_data(db_path):
    con = sqlite3.connect(db_path)
    cols = ", ".join(FEATURES)
    rows = con.execute(
        f"SELECT class, group_id, {cols} FROM training_patches").fetchall()
    con.close()
    y = np.array([MERGE.get(r[0], r[0]) for r in rows])
    groups = np.array([r[1] for r in rows])
    X = np.array([r[2:] for r in rows], dtype=np.float32)
    return X, y, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(HERE, 'training_data.db'))
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    X, y, groups = load_data(args.db)
    classes = sorted(set(y))
    cnt = Counter(y.tolist())
    print(f"Loaded {len(y)} patches in {len(set(groups))} groups, "
          f"{len(classes)} classes:")
    for c in classes:
        print(f"  {c:12s} {cnt[c]}")

    # RF is scale-invariant and is trained on RAW features (matches the map
    # model). The MDM baseline is distance-based, so it gets standardised input.
    Xs = StandardScaler().fit_transform(X)
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                               random_state=args.seed)
    rf = RandomForestClassifier(class_weight='balanced_subsample',
                                random_state=args.seed, n_jobs=-1, **RF_PARAMS)

    print(f"\n{args.folds}-fold GROUPED cross-validation (out-of-fold):")
    preds = {}
    preds['RF'] = cross_val_predict(rf, X, y, cv=skf, groups=groups, n_jobs=-1)
    preds['MDM'] = cross_val_predict(NearestCentroid(), Xs, y, cv=skf,
                                     groups=groups, n_jobs=-1)
    for tag, pred in preds.items():
        print(f"  {tag:>4}: accuracy {accuracy_score(y, pred)*100:5.1f}%   "
              f"macro-F1 {f1_score(y, pred, average='macro'):.3f}")

    out_csv = os.path.join(HERE, 'cv_metrics_v8.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'class', 'support', 'precision', 'recall', 'f1'])
        for tag, pred in preds.items():
            p, r, f1, sup = precision_recall_fscore_support(
                y, pred, labels=classes, zero_division=0)
            for i, c in enumerate(classes):
                w.writerow([tag, c, int(sup[i]), f'{p[i]:.4f}',
                            f'{r[i]:.4f}', f'{f1[i]:.4f}'])
    print(f"  wrote {out_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, tag in zip(axes, ('RF', 'MDM')):
        cm = confusion_matrix(y, preds[tag], labels=classes)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f"{tag} — acc {accuracy_score(y, preds[tag])*100:.1f}%",
                     weight='bold')
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cmn[i, j] > 0.5 else '#222', fontsize=9)
    fig.suptitle(f'Random Forest v8 vs minimum-distance-to-means '
                 f'({args.folds}-fold grouped CV)', fontsize=13, weight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(HERE, 'confusion_matrices_v8.png'), dpi=130,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {os.path.join(HERE, 'confusion_matrices_v8.png')}")

    rf.fit(X, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh([FEATURE_LABELS[i] for i in order][::-1], imp[order][::-1],
            color='#2c7fb8', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Gini importance')
    ax.set_title('Random Forest v8 feature importances', weight='bold')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'feature_importance_v8.png'), dpi=130,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {os.path.join(HERE, 'feature_importance_v8.png')}")

    # gzip-compressed so the (~250 MB raw) model fits under GitHub's 100 MB limit
    model_path = os.path.join(HERE, 'rf_v8_model.pkl.gz')
    with gzip.open(model_path, 'wb', compresslevel=6) as fh:
        pickle.dump({'model': rf, 'features': FEATURE_LABELS,
                     'feature_cols': FEATURES, 'classes': classes,
                     'rf_params': RF_PARAMS}, fh)
    print(f"  wrote {model_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
