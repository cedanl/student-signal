"""Tests for features module."""

import pandas as pd

from student_signal.features import convert_categorical_to_dummies, standardize_dataset


def test_convert_categorical_to_dummies() -> None:
    train = pd.DataFrame({"Dropout": [0, 1], "cat": ["a", "b"], "num": [1.0, 2.0]})
    pred = pd.DataFrame({"Dropout": [0, 0], "cat": ["a", "a"], "num": [3.0, 4.0]})
    train_out, pred_out = convert_categorical_to_dummies(train, pred, "Dropout")
    assert "cat" not in train_out.columns
    assert "num" in train_out.columns
    assert "Dropout" in train_out.columns
    assert train_out.shape[1] == pred_out.shape[1]


def test_standardize_dataset() -> None:
    train = pd.DataFrame({"Dropout": [0, 1, 0], "a": [10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0]})
    pred = pd.DataFrame({"Dropout": [1, 0], "a": [15.0, 25.0], "b": [1.5, 2.5]})
    train_out, pred_out, scaler = standardize_dataset(train, pred, "Dropout")
    assert "Dropout" in train_out.columns
    assert scaler is not None
    # Scaled features should be between 0 and 1
    non_dropout_cols = [c for c in train_out.columns if c != "Dropout"]
    for col in non_dropout_cols:
        assert train_out[col].min() >= -0.01
        assert train_out[col].max() <= 1.01
