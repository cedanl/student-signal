"""Model prediction functions: rank students by predicted dropout probability."""

from typing import Any

import pandas as pd


def _rank_predictions(
    predictions: pd.Series,
    studentnumbers: pd.DataFrame,
    studentnumber_column: str,
) -> pd.DataFrame:
    """Create a ranked DataFrame from model predictions.

    Args:
        predictions: Series of predicted dropout probabilities.
        studentnumbers: DataFrame with the student number column.
        studentnumber_column: Name of the student number column.

    Returns:
        DataFrame with ranking, student number, and prediction columns.
    """
    pred_data = pd.DataFrame({"voorspelling": predictions})
    pred_data = pd.concat([pred_data, studentnumbers.reset_index(drop=True)], axis=1)
    pred_data["ranking"] = pred_data["voorspelling"].rank(method="dense", ascending=False)
    pred_data = pred_data.sort_values(by=["voorspelling"], ascending=False).reset_index(drop=True)
    return pred_data[["ranking", studentnumber_column, "voorspelling"]]


def predict_model(
    model: Any,
    pred_df: pd.DataFrame,
    dropout_column: str,
    studentnumber_column: str,
) -> pd.DataFrame:
    """Generate ranked dropout predictions for any sklearn-compatible model.

    Automatically uses predict_proba for classifiers and predict for regressors.

    Args:
        model: Fitted model with .predict() or .predict_proba().
        pred_df: Prediction DataFrame with features and optionally a dropout column.
        dropout_column: Name of the target column (dropped before predicting if present).
        studentnumber_column: Name of the student ID column.

    Returns:
        DataFrame with students ranked by predicted dropout probability.
    """
    X_pred = (
        pred_df.drop(columns=[dropout_column])
        if dropout_column in pred_df.columns
        else pred_df.drop(columns=[studentnumber_column], errors="ignore")
    )
    studentnumbers = pred_df[[studentnumber_column]]

    if hasattr(model, "predict_proba"):
        predictions = pd.Series(model.predict_proba(X_pred)[:, 1], index=X_pred.index)
    else:
        predictions = pd.Series(model.predict(X_pred), index=X_pred.index)

    return _rank_predictions(predictions, studentnumbers, studentnumber_column)
