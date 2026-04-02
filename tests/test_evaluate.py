"""Tests for evaluate module."""

from student_signal.evaluate import get_stoplight_evaluation, load_settings


def test_load_settings_default() -> None:
    settings = load_settings()
    assert "imputation" in settings
    assert "hyperparameters" in settings
    assert "evaluation" in settings
    assert settings["imputation"]["n_neighbors"] == 5
    assert settings["hyperparameters"]["random_seed"] == 42


def test_stoplight_green() -> None:
    emoji, status, _ = get_stoplight_evaluation(50.0, 45.0)
    assert emoji == "🟢"
    assert status == "Betrouwbaar"


def test_stoplight_yellow() -> None:
    emoji, status, _ = get_stoplight_evaluation(35.0, 35.0)
    assert emoji == "🟡"


def test_stoplight_red() -> None:
    emoji, status, _ = get_stoplight_evaluation(20.0, 25.0)
    assert emoji == "🔴"
    assert status == "Niet bruikbaar"
