# Drone CV — Pasture Species Classifier (reproducible package)

Self-contained Random Forest classifiers for multispectral drone imagery of
pasture. **Training needs no imagery** — the labelled data ships as a SQLite
database. Making a full classified map additionally needs the orthophotos.

The current model is **v8**; **v2** is kept as the earlier baseline.

## What's included

| File | Description |
|------|-------------|
| `training_data.db` | 21,007 labelled ~30 cm patches (KML placemarks + MRK transects), with 15 features each, a `group_id`, and centre lat/lon. |
| `train_rf_v8.py` | **Current** trainer: 15 features, 5 classes, honest grouped cross-validation. Writes plots, metrics, and the model. |
| `generate_map.py` | Classify whole orthophotos with the trained v8 model and render a Leaflet web map. **Needs imagery.** |
| `rf_v8_model.pkl.gz` | The fitted v8 Random Forest (gzip-compressed; ~74 MB). |
| `train_rf_v2.py` | Legacy v2 trainer (9 spectral features, non-grouped CV). |
| `requirements.txt` | Python dependencies. |

## The v8 model

- **15 features per patch:**
  - 9 spectral — indices `ndvi, ndre, gndvi, gri, sr` + raw bands `red, green, nir, rededge`
  - 6 neighborhood stats — std/mean of NDVI/NIR/GRI over 3×3 / 5×5 patch windows:
    `ndvi_std3, nir_std3, gri_std3, ndvi_mean5, nir_mean5, ndvi_std5`
- **5 classes:** Leguminosa, Graminea, Other, Bare Soil, Dry Grass.
  The database stores `Water Hole` separately (185 patches); the trainer folds it
  into `Bare Soil` (both are non-vegetated ground) — see `MERGE` in `train_rf_v8.py`.
- **Grouped cross-validation** (`StratifiedGroupKFold` on `group_id`): all patches
  from one placemark / MRK point stay in the same fold, so the score isn't inflated
  by near-duplicate neighbours. This is the honest number.

Class counts (after the Water Hole→Bare Soil merge): Graminea 8378 · Dry Grass
5934 · Leguminosa 4440 · Other 1663 · Bare Soil 592.

## Train

```bash
pip install -r requirements.txt   # numpy, scikit-learn, matplotlib suffice for training
python train_rf_v8.py
```

Expected console output (4-fold grouped CV, out-of-fold):

```
   RF: accuracy  72.1%   macro-F1 0.691
  MDM: accuracy  48.9%   macro-F1 0.459
```

Writes `confusion_matrices_v8.png`, `feature_importance_v8.png`,
`cv_metrics_v8.csv`, and `rf_v8_model.pkl.gz`.
Flags: `--folds 4`, `--seed 42`, `--db path/to.db`.

## Generate a map (needs orthophotos)

`generate_map.py` recomputes the same 15 features from raw imagery, classifies
every patch, and writes per-image GeoTIFFs + a `classification_map.html` with the
training points overlaid. Imagery must be a 5-band orthophoto
(1=Red, 2=Green, 3=NIR, 4=RedEdge, 5=alpha/mask).

```bash
python generate_map.py \
    --imagery "/path/to/orthophotos/*/odm_orthophoto.tif" \
    --outdir map_output
```

Flags: `--patch-size 30` (cm), `--min-valid-frac 0.5`, `--opacity 0.8`,
`--model`, `--db`. Output: `map_output/classification_map.html`.

## Using the trained model on new patches

```python
import gzip, pickle, numpy as np
b = pickle.load(gzip.open('rf_v8_model.pkl.gz', 'rb'))
# 15 features in b['feature_cols'] order:
# ndvi ndre gndvi gri sr red green nir rededge ndvi_std3 nir_std3 gri_std3 ndvi_mean5 nir_mean5 ndvi_std5
x = np.array([[0.27,0.12,0.40,-0.15,1.73,0.025,0.018,0.043,0.034,
               0.05,0.004,0.03,0.27,0.043,0.05]])
print(b['model'].predict(x))   # -> e.g. ['Leguminosa']   (RF uses raw, unscaled features)
```

## Notes & caveats

- **Headline 72.1% is the honest grouped number.** Non-grouped CV looks higher
  but leaks near-duplicate patches across folds.
- **MRK transects are spatially dense** and grouped per-point, so their per-class
  recalls are still mildly optimistic; a per-transect-grouped run would be stricter.
- **Map class areas partly reflect the training prior**, not only ground truth —
  read them as relative, and sanity-check against the satellite layer.
- The dominant residual confusion is **Leguminosa ↔ Graminea** (spectral overlap
  at 5-band / 5 cm). Texture features (GLCM/LBP) were tested and added only ~1 pp,
  so they're intentionally left out of this 15-feature production model.

### Legacy v2

`train_rf_v2.py` reads only the 9 spectral columns and uses non-grouped CV. Its
originally published numbers (~58.8%) were on the earlier 9-feature, KML-only,
8,249-patch database; run against this expanded database it will train on more
data (and the extra columns are simply ignored).
