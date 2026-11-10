"""Tests for the public API: prepare() and rank()."""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from student_signal import prepare, rank
from student_signal.modeling.predict import predict_model

FRAME_NAMES = (
    "X_train",
    "X_pred",
    "X_train_scaled",
    "X_pred_scaled",
    "train_df",
    "train_df_scaled",
    "pred_df",
    "pred_df_scaled",
)


def _train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Studentnummer": [101, 102, 103, 104, 105, 106],
            "Dropout": [0, 0, 0, 1, 1, 1],
            "verzuim": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
            "richting": ["a", "b", "a", "b", "a", "b"],
        }
    )


def _pred_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Studentnummer": [201, 202, 203],
            "Dropout": [0, 0, 0],
            "verzuim": [9.0, 1.0, 5.0],
            "richting": ["a", "a", "b"],
        }
    )


def _fit_on_verzuim(prepared: object) -> LinearRegression:
    return LinearRegression().fit(prepared.X_train, prepared.y_train)


def test_prepare_keeps_id_out_of_features() -> None:
    prepared = prepare(_train_df(), _pred_df())
    for name in FRAME_NAMES:
        assert "Studentnummer" not in getattr(prepared, name).columns, name
    assert prepared.train_ids.tolist() == [101, 102, 103, 104, 105, 106]
    assert prepared.pred_ids.tolist() == [201, 202, 203]


def test_prepare_result_does_not_depend_on_id_values() -> None:
    other_ids = _train_df().assign(Studentnummer=[9, 8, 7, 6, 5, 4])
    first = prepare(_train_df(), _pred_df())
    second = prepare(other_ids, _pred_df())
    pd.testing.assert_frame_equal(first.X_train_scaled, second.X_train_scaled)
    pd.testing.assert_frame_equal(first.X_pred_scaled, second.X_pred_scaled)


def test_prepare_raises_when_id_column_is_missing() -> None:
    with pytest.raises(ValueError, match="id_col 'Studentnummer' not found in pred_df"):
        prepare(_train_df(), _pred_df().drop(columns=["Studentnummer"]))


def test_rank_returns_text_ids_unchanged() -> None:
    train = _train_df().assign(Studentnummer=["a1", "a2", "a3", "a4", "a5", "a6"])
    pred = _pred_df().assign(Studentnummer=["jan", "piet", "klaas"])
    prepared = prepare(train, pred)
    ranked = rank(_fit_on_verzuim(prepared), prepared)
    assert ranked["Studentnummer"].tolist() == ["jan", "klaas", "piet"]


@pytest.mark.parametrize("use_scaled", [False, True])
def test_rank_matches_ids_by_position_when_index_has_gaps(use_scaled: bool) -> None:
    pred = _pred_df()
    pred.index = [0, 5, 9]
    prepared = prepare(_train_df(), pred)
    X_train = prepared.X_train_scaled if use_scaled else prepared.X_train
    model = LinearRegression().fit(X_train, prepared.y_train)

    ranked = rank(model, prepared, use_scaled=use_scaled)

    assert len(ranked) == 3
    assert ranked["Studentnummer"].tolist() == [201, 203, 202]
    assert ranked["voorspelling"].notna().all()


def test_predict_model_drops_id_and_target() -> None:
    prepared = prepare(_train_df(), _pred_df())
    model = _fit_on_verzuim(prepared)
    pred_with_id = prepared.pred_df.assign(Studentnummer=prepared.pred_ids.values)

    ranked = predict_model(model, pred_with_id, "Dropout", "Studentnummer")

    assert ranked["Studentnummer"].tolist() == [201, 203, 202]
