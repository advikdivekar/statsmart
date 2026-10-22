"""End-to-end pipeline: raw CSVs -> cross-validated models -> blended submission.

Run:  .venv/bin/python main.py
Output: outputs/submissions/submission_v3.csv (+ figures and cv_results.csv)
(v1's outputs/submissions/submission.csv is kept untouched: probe.py builds on it.
 solution_v2.py regenerates submission_v2.csv on its own.)
"""
import warnings

import numpy as np
import pandas as pd

from src import config, data, ensemble, evaluate, features, models, plots

warnings.filterwarnings("ignore")


def main():
    for folder in (config.SUBMISSION_DIR, config.FIGURE_DIR, config.MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Data: clean and engineer features on train+test stacked (no target is used).
    train_raw, test_raw = data.load_raw()
    full = features.add_features(data.clean(pd.concat([train_raw, test_raw], ignore_index=True)))
    train, test = features.split_train_test(full, len(train_raw))
    cols = features.feature_columns(full)
    X_train = features.to_category_dtype(train[cols])
    X_test = features.to_category_dtype(test[cols])
    y = train[config.TARGET]

    # 2. Cross-validate every model and keep its out-of-fold predictions.
    all_models = models.get_models()
    oof, scores = {}, {}
    for name, model in all_models.items():
        mean_rmse, std_rmse, oof[name] = evaluate.cross_validate(model, X_train, y)
        scores[name] = mean_rmse
        print(f"{name:20s} CV RMSE {mean_rmse:8.1f} (+- {std_rmse:.1f})")

    # 3. Blend weights from the out-of-fold predictions.
    weights = ensemble.blend_weights(oof, y)
    blend_oof = ensemble.apply_weights(oof, weights)
    scores["blend"] = evaluate.rmse(y, blend_oof)
    print("blend weights:", {k: round(v, 3) for k, v in weights.items()})
    print(f"{'blend':20s} OOF RMSE {scores['blend']:8.1f}")

    # 4. Refit on all training data, predict test, apply the blend.
    test_predictions = {}
    for name, model in all_models.items():
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
    submission.to_csv(config.SUBMISSION_V3_CSV, index=False)

    pd.Series(scores, name="cv_rmse").to_csv(config.OUTPUT_DIR / "cv_results.csv")
    plots.save_model_comparison(scores)
    plots.save_actual_vs_predicted(y, blend_oof)
    plots.save_id_number_effect(models.PerOutletLinearTimesIDNumber().fit(X_train, y), X_train, y)
    plots.save_sales_vs_mrp_by_outlet_type(X_train, y)
    print(f"saved {config.SUBMISSION_V3_CSV}")


if __name__ == "__main__":
    main()
