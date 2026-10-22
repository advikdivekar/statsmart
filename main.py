"""End-to-end pipeline: raw CSVs -> cross-validated models -> blended submission (v5).

Run:  .venv/bin/python main.py
Output: outputs/submissions/submission_v5.csv (+ figures and cv_results.csv)
(Earlier versions regenerate their own CSVs: solution.py for v1, solution_v2.py, solution_v3.py
 and solution_v4.py. probe.py builds on v1's outputs/submissions/submission.csv.)

Nothing from the test file is used for fitting: the models read only the item id, Outlet_Identifier
and Item_MRP, built row by row (features.model_frame); v5's true item prices come from train sales;
test rows only reach predict().
"""
import warnings

import numpy as np
import pandas as pd

from src import config, data, ensemble, evaluate, features, models, plots

warnings.filterwarnings("ignore")


def id_effect(fitted_models, weights, X):
    """The blend's relative effect of the id number: predictions for one typical row (most common
    outlet, median MRP, no known item price) at every number 1..60, divided by their mean."""
    grid = pd.DataFrame({
        "Item_Identifier": "",
        "Outlet_Identifier": X["Outlet_Identifier"].mode().iat[0],
        "Item_MRP": X["Item_MRP"].median(),
        "ID_Number": np.arange(1, 61),
    })
    pred = ensemble.apply_weights({n: m.predict(grid) for n, m in fitted_models.items()}, weights)
    return pd.Series(pred / pred.mean() - 1, index=grid["ID_Number"])


def main():
    for folder in (config.SUBMISSION_DIR, config.FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Data: the per-row model columns, built for train and test separately.
    train_raw, test_raw = data.load_raw()
    X_train, X_test = features.model_frame(train_raw), features.model_frame(test_raw)
    y = train_raw[config.TARGET]

    # 2. Cross-validate the models of v3, v4 and v5 (v5 is submitted; the others are for comparison).
    pools = {"v3": models.get_v3_blend_models(), "v4": models.get_v4_models(), "v5": models.get_v5_models()}
    oof, scores = {}, {}
    for pool in pools.values():
        for name, model in pool.items():
            mean_rmse, std_rmse, oof[name] = evaluate.cross_validate(model, X_train, y)
            scores[name] = mean_rmse
            print(f"{name:26s} CV RMSE {mean_rmse:8.1f} (+- {std_rmse:.1f})")

    # 3. Blend each version's models with NNLS weights from the out-of-fold predictions.
    blends = {}
    for version, pool in pools.items():
        pool_oof = {n: oof[n] for n in pool}
        version_weights = ensemble.blend_weights(pool_oof, y)
        blends[version] = (version_weights, ensemble.apply_weights(pool_oof, version_weights))
        scores[f"{version} blend"] = evaluate.rmse(y, blends[version][1])
    weights, blend_oof = blends["v5"]
    print("v5 blend weights:", {k: round(float(v), 3) for k, v in weights.items()})
    for version in pools:
        print(f"{version + ' blend':26s} OOF RMSE {scores[version + ' blend']:8.1f}")

    # 4. Refit v5's models on all training data, predict test, apply the blend.
    v5_models = pools["v5"]
    test_predictions = {}
    for name, model in v5_models.items():
        if weights[name] > 0:
            test_predictions[name] = model.fit(X_train, y).predict(X_test)
    final = np.clip(config.PREDICTION_SCALE * ensemble.apply_weights(test_predictions, weights), 0, None)
    print(f"prediction scale: {config.PREDICTION_SCALE}")

    # 5. Write the submission in the exact sample format.
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV)
    submission = test_raw[config.ID_COLS].copy()
    submission[config.TARGET] = final
    assert list(submission.columns) == list(sample.columns)
    assert submission[config.ID_COLS].equals(sample[config.ID_COLS]), "row order differs from sample"
    assert len(submission) == len(sample) and submission[config.TARGET].notna().all()
    submission.to_csv(config.SUBMISSION_V5_CSV, index=False)

    pd.Series(scores, name="cv_rmse").to_csv(config.OUTPUT_DIR / "cv_results.csv")
    plots.save_model_comparison(scores)
    plots.save_actual_vs_predicted(y, blend_oof)
    v5_fitted = {n: m for n, m in v5_models.items() if weights[n] > 0}
    plots.save_id_number_effect(models.PerOutletLinearTimesIDNumber().fit(X_train, y), X_train, y,
                                id_effect(v5_fitted, weights, X_train))
    plots.save_sales_vs_mrp_by_outlet_type(X_train, y)
    print(f"saved {config.SUBMISSION_V5_CSV}")


if __name__ == "__main__":
    main()
