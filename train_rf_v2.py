"""
Random Forest v2 — standalone, reproducible trainer.

Trains a Random Forest pasture classifier (and a minimum-distance-to-means
baseline) on the bundled training database and reports a cross-validated
comparison. Self-contained: needs only the .db file in this folder, no drone
imagery.

Each row in training_data.db is one ~30 cm ground-truth patch: its class label
plus 9 spectral features (5 vegetation indices + 4 raw band reflectances).

Run:
    pip install -r requirements.txt
    python train_rf_v2.py

Outputs (written next to this script):
    confusion_matrices.png   RF vs MDM, 5-fold cross-validation
    feature_importance.png   which features drive the Random Forest
    cv_metrics.csv           per-class precision / recall / F1
    rf_v2_model.pkl          the trained Random Forest (+ scaler + labels)
"""
import argparse
import csv
import os
import pickle
import sqlite3

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt

FEATURES = ['ndvi', 'ndre', 'gndvi', 'gri', 'sr', 'red', 'green', 'nir', 'rededge']
FEATURE_LABELS = ['NDVI', 'NDRE', 'GNDVI', 'GRI', 'SR', 'Red', 'Green', 'NIR', 'RedEdge']
HERE = os.path.dirname(os.path.abspath(__file__))


def load_data(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cols = ", ".join(FEATURES)
    rows = cur.execute(f"SELECT class, {cols} FROM training_patches").fetchall()
    con.close()
    y = np.array([r[0] for r in rows])
    X = np.array([r[1:] for r in rows], dtype=np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(HERE, 'training_data.db'))
    ap.add_argument('--trees', type=int, default=300)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    X, y = load_data(args.db)
    classes = sorted(set(y))
    print(f"Loaded {len(y)} training patches, {len(classes)} classes:")
    for c in classes:
        print(f"  {c:12s} {int((y == c).sum())}")

    # Standardise features (so the distance-based baseline is fair). RF is
    # scale-invariant, so this only matters for the MDM baseline.
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)
    models = {
        'RF':  RandomForestClassifier(
                   n_estimators=args.trees, class_weight='balanced_subsample',
                   random_state=args.seed, n_jobs=-1),
        'MDM': NearestCentroid(),
    }

    print(f"\n{args.folds}-fold cross-validation (out-of-fold predictions):")
    preds = {}
    for tag, est in models.items():
        pred = cross_val_predict(est, Xs, y, cv=skf, n_jobs=-1)
        preds[tag] = pred
        print(f"  {tag:>4}: accuracy {accuracy_score(y, pred)*100:5.1f}%   "
              f"macro-F1 {f1_score(y, pred, average='macro'):.3f}")

    # Per-class metrics CSV
    out_csv = os.path.join(HERE, 'cv_metrics.csv')
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

    # Confusion matrices
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
    fig.suptitle(f'Random Forest vs minimum-distance-to-means '
                 f'({args.folds}-fold CV)', fontsize=13, weight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(HERE, 'confusion_matrices.png'), dpi=130,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {os.path.join(HERE, 'confusion_matrices.png')}")

    # Fit final RF on all data, save it + feature importances
    rf = models['RF'].fit(Xs, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([FEATURE_LABELS[i] for i in order][::-1], imp[order][::-1],
            color='#2c7fb8', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Gini importance'); ax.set_title('Random Forest feature importances',
                                                    weight='bold')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'feature_importance.png'), dpi=130,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {os.path.join(HERE, 'feature_importance.png')}")

    with open(os.path.join(HERE, 'rf_v2_model.pkl'), 'wb') as fh:
        pickle.dump({'model': rf, 'scaler': scaler, 'features': FEATURE_LABELS,
                     'classes': classes}, fh)
    print(f"  wrote {os.path.join(HERE, 'rf_v2_model.pkl')}")
    print("\nDone.")


if __name__ == '__main__':
    main()
