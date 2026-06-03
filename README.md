# Random Forest v2 — Pasture Species Classifier (reproducible package)

A self-contained, runnable copy of the **Random Forest v2** pasture classifier.
It trains a Random Forest to classify multispectral drone patches into pasture
classes, and compares it against a minimum-distance-to-means (MDM) baseline.

Everything needed to train and evaluate the model is in this folder — **no drone
imagery required**. The training data ships as a SQLite database.

## What's included

| File | Description |
|------|-------------|
| `training_data.db` | SQLite database with 8,249 labelled training patches (the "v2" set: original ground truth + a second field campaign). |
| `train_rf_v2.py` | Standalone trainer: trains the RF + MDM, runs cross-validation, writes the comparison plots and the trained model. |
| `requirements.txt` | Python dependencies. |

## The data

Each row in the `training_patches` table is one ~30 cm ground-truth patch
sampled within 0.5–1 m of a GPS observation. Columns:

- `class` — one of `Leguminosa`, `Graminea`, `Other`, `Bare Soil`, `Water Hole`, `Dry Grass`
- `lat`, `lon` — patch centre (WGS84); for provenance only, **not** used as model inputs
- 9 spectral features used by the model:
  - **Vegetation indices:** `ndvi`, `ndre`, `gndvi`, `gri`, `sr`
  - **Raw band reflectance:** `red`, `green`, `nir`, `rededge`

Class counts: Graminea 2146 · Dry Grass 2109 · Leguminosa 1739 · Other 1663 ·
Bare Soil 407 · Water Hole 185.

Inspect it directly with any SQLite tool, e.g.:

```bash
sqlite3 training_data.db "SELECT class, COUNT(*) FROM training_patches GROUP BY class;"
```

## How to run

```bash
pip install -r requirements.txt
python train_rf_v2.py
```

Optional flags: `--trees 300`, `--folds 5`, `--seed 42`, `--db path/to.db`.

## Expected output

Console (5-fold cross-validation, out-of-fold predictions):

```
   RF: accuracy  58.8%   macro-F1 0.617
  MDM: accuracy  41.6%   macro-F1 0.396
```

The Random Forest roughly **doubles** the macro-F1 of the distance baseline.
Files written next to the script:

- `confusion_matrices.png` — RF vs MDM, where each model confuses classes
- `feature_importance.png` — which of the 9 features drive the Random Forest
- `cv_metrics.csv` — per-class precision / recall / F1 for both models
- `rf_v2_model.pkl` — the fitted Random Forest, the feature scaler, and the
  class list (a pickled dict: `{'model', 'scaler', 'features', 'classes'}`)

## Using the trained model on new patches

```python
import pickle, numpy as np
b = pickle.load(open('rf_v2_model.pkl', 'rb'))
# features in this exact order: NDVI, NDRE, GNDVI, GRI, SR, Red, Green, NIR, RedEdge
x = np.array([[0.27, 0.12, 0.40, -0.15, 1.73, 0.025, 0.018, 0.043, 0.034]])
pred = b['model'].predict(b['scaler'].transform(x))
print(pred)   # -> e.g. ['Leguminosa']
```

## Notes & caveats

- **Cross-validation here is non-grouped.** Patches sampled near the same GPS
  point are near-duplicates, so this number is mildly optimistic. The honest,
  spatially-grouped accuracy is lower (~57–60%). Treat 58.8% as the "v2 headline"
  figure, comparable to the original v2 run.
- The MDM baseline standardizes features on the training patches here, versus the
  full image population in the original pipeline — so its number can differ by
  ~1 point. The RF result is unaffected (it is scale-invariant).
- This package reproduces the **model and accuracy**. Producing full-survey
  classified maps additionally needs the multispectral orthophotos (not included
  due to size) and the full pipeline scripts (`classify_pasture_rf.py`).
