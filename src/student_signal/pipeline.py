"""Public API: prepare(), rank(), evaluate() — the three entry points for library consumers."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from student_signal.dataset import impute_missing_values, remove_single_value_columns
from student_signal.evaluate import generate_stoplight_evaluation, load_settings
from student_signal.features import convert_categorical_to_dummies, standardize_dataset
from student_signal.modeling.predict import _rank_predictions


@dataclass
class PreparedData:
    """Output of prepare(). Contains everything needed to train and predict.

    Attributes:
        X_train: Feature matrix for training (unscaled).
        y_train: Target vector for training.
        X_pred: Feature matrix for prediction (unscaled).
        X_train_scaled: Feature matrix for training (scaled).
        X_pred_scaled: Feature matrix for prediction (scaled, same scale as train).
        imputer: Fitted KNNImputer. Serialise this alongside your model.
        scaler: Fitted MinMaxScaler. Serialise this alongside your model.
        train_df: Full training DataFrame (features + target, unscaled).
        train_df_scaled: Full training DataFrame (features + target, scaled).
        pred_df: Full prediction DataFrame (features + target column, unscaled).
        pred_df_scaled: Full prediction DataFrame (features + target column, scaled).
        target_col: Name of the target column.
        id_col: Name of the student ID column.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_pred: pd.DataFrame
    X_train_scaled: pd.DataFrame
    X_pred_scaled: pd.DataFrame
    imputer: KNNImputer
    scaler: MinMaxScaler
    train_df: pd.DataFrame
    train_df_scaled: pd.DataFrame
    pred_df: pd.DataFrame
    pred_df_scaled: pd.DataFrame
    target_col: str
    id_col: str


def prepare(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_col: str = "Dropout",
    id_col: str = "Studentnummer",
    config: dict | str | None = None,
) -> PreparedData:
    """Clean and transform raw student data into train/predict matrices.

    This function applies the full preprocessing pipeline:
    - Missing value imputation (KNN, fit on train only)
    - Removal of constant-value columns
    - One-hot encoding of categorical columns
    - Min-max scaling (fit on train only)

    The fitted scaler is returned inside PreparedData so it can be serialised
    alongside any trained model and reused consistently at predict time.

    Args:
        train_df: Raw training DataFrame including target column.
        pred_df: Raw prediction DataFrame including target column (may be zeros/NaN).
        target_col: Name of the dropout/target column. Default: 'Dropout'.
        id_col: Name of the student ID column. Default: 'Studentnummer'.
        config: Optional config overrides. Pass a dict to override specific
            library defaults, or a path string to load a custom YAML entirely.

    Returns:
        PreparedData with all matrices needed for training and prediction.
    """
    settings = load_settings(
        config_file=config if isinstance(config, str) else None,
        overrides=config if isinstance(config, dict) else None,
    )
    n_neighbors = settings.get("imputation", {}).get("n_neighbors", 5)

    train_df, pred_df, imputer = impute_missing_values(train_df, pred_df, n_neighbors=n_neighbors)
    train_df, pred_df = remove_single_value_columns(train_df, pred_df)
    train_df, pred_df = convert_categorical_to_dummies(train_df, pred_df, target_col)
    train_df_scaled, pred_df_scaled, scaler = standardize_dataset(train_df, pred_df, target_col)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_pred = pred_df.drop(columns=[target_col])
    X_train_scaled = train_df_scaled.drop(columns=[target_col])
    X_pred_scaled = pred_df_scaled.drop(columns=[target_col])

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_pred=X_pred,
        X_train_scaled=X_train_scaled,
        X_pred_scaled=X_pred_scaled,
        imputer=imputer,
        scaler=scaler,
        train_df=train_df,
        train_df_scaled=train_df_scaled,
        pred_df=pred_df,
        pred_df_scaled=pred_df_scaled,
        target_col=target_col,
        id_col=id_col,
    )


def rank(
    model: Any,
    prepared: PreparedData,
    use_scaled: bool = False,
) -> pd.DataFrame:
    """Rank students by predicted dropout probability.

    Args:
        model: Any fitted sklearn-compatible estimator with .predict() or .predict_proba().
        prepared: Output of prepare().
        use_scaled: If True, predict on scaled features (for Lasso/SVM).
            If False (default), predict on unscaled features (for tree-based models).

    Returns:
        DataFrame with columns [rank, <id_col>, score], sorted highest risk first.
    """
    X = prepared.X_pred_scaled if use_scaled else prepared.X_pred
    studentnumbers = prepared.pred_df[[prepared.id_col]]

    if hasattr(model, "predict_proba"):
        predictions = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    else:
        predictions = pd.Series(model.predict(X), index=X.index)

    return _rank_predictions(predictions, studentnumbers, prepared.id_col)


def evaluate(
    models: dict[str, tuple[Any, bool]],
    prepared: PreparedData,
    invite_pct: int = 20,
    reports_dir: str | None = None,
) -> dict:
    """Evaluate model quality using the stoplight methodology.

    Args:
        models: Dict mapping display names to (fitted_model, use_scaled) tuples.
            Set use_scaled=True for models trained on scaled data (Lasso, SVM).
            Set use_scaled=False for models trained on unscaled data (tree-based).
        prepared: Output of prepare().
        invite_pct: Invitation percentage threshold for stoplight evaluation.
        reports_dir: If provided, save report files and figures to this directory.

    Returns:
        Dict with evaluation metrics per model plus an 'Aanbeveling' key.
    """
    from pathlib import Path

    model_predictions = {
        name: (
            prepared.train_df_scaled if use_scaled else prepared.train_df,
            model,
        )
        for name, (model, use_scaled) in models.items()
    }

    kwargs: dict[str, Any] = {
        "model_predictions": model_predictions,
        "invite_pct": invite_pct,
        "dropout_column": prepared.target_col,
    }
    if reports_dir is not None:
        kwargs["reports_dir"] = Path(reports_dir)

    return generate_stoplight_evaluation(**kwargs)
